import copy
import typing

import numpy as np
import revlib
import torch
import torch.nn.functional
import torch.utils.data
from deepspeed.runtime import lr_schedules

from src.dataclass import Context
from src.optimizers.build import build_optimizer

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def orthonormal(inp: typing.Union[torch.Tensor, torch.nn.Parameter, typing.List[int]], gain: float):
    original_input = inp
    if isinstance(inp, list):
        inp = torch.zeros(inp)
    if isinstance(inp, torch.nn.Parameter):
        inp = inp.data
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    a = torch.rand(flat_shape)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    inp.copy_((u if u.shape == flat_shape else v).reshape(inp.shape).mul(gain).to(device=inp.device, dtype=inp.dtype))
    if isinstance(original_input, list):
        return torch.nn.Parameter(inp)
    return original_input


def norm(out: torch.Tensor) -> torch.Tensor:
    out = out - out.mean(1, keepdim=True)
    return out / (torch.norm(out, 2, 1, True) * out.size(1) ** -0.5 + 1e-5)


def conv(inp: torch.Tensor, weight: torch.Tensor, groups: int, use_pad: bool) -> torch.Tensor:
    if use_pad and weight.size()[-1] - 1 > 0:
        inp = torch.nn.functional.pad(inp, (weight.size()[-1] - 1, 0))
    return torch.nn.functional.conv1d(inp, weight, groups=groups)


def drop_conv(inp: torch.Tensor, weight: torch.Tensor, p: float, train: bool, groups: int, pad: bool) -> torch.Tensor:
    batch, features, sequence = inp.size()
    if 0 < p < 1:
        if train:
            mask = torch.randn((features,), device=inp.device, dtype=inp.dtype) < p
            inp = torch.masked_select(inp, mask.view(1, -1, 1)).view(batch, -1, sequence)
            weight = torch.masked_select(weight, mask.view(-1, 1, 1)).view(-1, weight.size(1), weight.size(2))
        elif torch.numel(inp) > torch.numel(weight):
            weight = weight * p
        else:
            inp = inp * p
    return conv(inp, weight, groups, pad)


def moe(inp: torch.Tensor, w: typing.List[torch.Tensor],
        gate: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    out = torch.nn.functional.conv1d(inp, gate)
    gates = torch.nn.functional.softmax(out, dim=1)
    one_hot = torch.nn.functional.one_hot(torch.argmax(out, dim=1), num_classes=out.shape[1])
    gumbel = one_hot.transpose(1, 2) - gates.detach() + gates
    one_hot = one_hot.to(dtype=torch.bool)
    inp_t = inp.transpose(1, 2)
    batch, features, sequence = inp.size()
    out = torch.empty((batch * sequence, w[0].size(1)), device=inp.device, dtype=inp.dtype)
    for expert, g, param in zip(one_hot.unbind(-1), gumbel.unbind(1), w):
        tmp = torch.masked_select(inp_t * g.unsqueeze(2), expert.unsqueeze(2)).view(-1, features).mm(param)
        out = out.masked_scatter(expert.view(-1, 1), tmp)
    loss = torch.sum(torch.mean(gates, dim=(0, 2)) * torch.mean(one_hot.float(), dim=(0, 1)))
    return loss, out.view(batch, sequence, -1).transpose(1, 2)


def moe_check(inp: torch.Tensor, w_gate: torch.Tensor, w: typing.List[torch.Tensor], dropout_probability: float,
              training: bool, groups: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    if w:
        return moe(inp, w, w_gate)
    return (torch.zeros([1], device=inp.device, dtype=inp.dtype),
            drop_conv(inp, w_gate, dropout_probability, training, groups, False))


def split_norm(inp: torch.Tensor) -> torch.Tensor:
    scale0, scale1, shift = inp.chunk(3, 1)
    return norm(scale0 * scale1 + shift)


@torch.jit.script
def linear_attention(inp: torch.Tensor, divisor: torch.Tensor, w0_gate: torch.Tensor,
                     w0: typing.List[torch.Tensor], w1: torch.Tensor, w2_gate: torch.Tensor,
                     w2: typing.List[torch.Tensor], input_cache: torch.Tensor, cumsum_cache: torch.Tensor,
                     init_scale: float, bottleneck_group: int, dropout_probability: float, training: bool,
                     caching: bool, idx: int
                     ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_size = w1.size(2)
    pad = True
    if not training and caching:
        if idx - 1 > kernel_size and inp.size(2) == 1:
            pad = False
            inp = torch.cat([input_cache, inp], -1)
        input_cache = inp[:, :, -kernel_size + 1:].detach()
    loss0, inp = moe_check(inp, w0_gate, w0, dropout_probability, training, 1)
    depth, scale, shift = inp.chunk(3, 1)
    cum = depth.cumsum(-1)
    if not training and caching:
        cum = cum + cumsum_cache
        scale = scale[:, :, -1:]
        shift = shift[:, :, -1:]
        cum = cum[:, :, -1:]
        if idx - 1 > kernel_size:
            cumsum_cache = cum.detach()
    inp = torch.nn.functional.leaky_relu(norm(cum / divisor * scale + shift), 0.02)
    inp = drop_conv(inp, w1, dropout_probability, training, bottleneck_group, pad)
    inp = torch.nn.functional.leaky_relu(split_norm(inp), 0.02)
    loss1, inp = moe_check(inp, w2_gate, w2, dropout_probability, training, 1)
    return loss0, loss1, input_cache, cumsum_cache, inp * init_scale


def get_coupling(beta_tmp: float):
    @torch.jit.script
    def momentum_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor, beta: float) -> torch.Tensor:
        return other_stream * beta + fn_out

    @torch.jit.script
    def momentum_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor, beta: float) -> torch.Tensor:
        return (output - fn_out) / beta

    def _wrapped_momentum_coupling_forward(x, y):
        return momentum_coupling_forward(x, y, beta_tmp)

    def _wrapped_momentum_coupling_inverse(x, y):
        return momentum_coupling_inverse(x, y, beta_tmp)

    return _wrapped_momentum_coupling_forward, _wrapped_momentum_coupling_inverse


def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv1d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)


class Trainer(torch.nn.Module):
    def __init__(self, ctx: Context, model: torch.nn.Module):
        super(Trainer, self).__init__()
        self.ctx = ctx
        self.model = model
        self.optimizer = build_optimizer(ctx, self.model.parameters())
        self.scheduler = lr_schedules.OneCycle(self.optimizer,
                                               ctx.optimizer.one_cycle.cycle_min_lr,
                                               ctx.optimizer.one_cycle.cycle_max_lr,
                                               ctx.optimizer.one_cycle.decay_lr_rate,
                                               ctx.optimizer.one_cycle.cycle_first_step_size,
                                               ctx.optimizer.one_cycle.cycle_second_step_size,
                                               ctx.optimizer.one_cycle.cycle_first_stair_count,
                                               ctx.optimizer.one_cycle.cycle_second_stair_count,
                                               ctx.optimizer.one_cycle.decay_step_size,
                                               ctx.optimizer.one_cycle.cycle_momentum,
                                               ctx.optimizer.one_cycle.cycle_min_mom,
                                               ctx.optimizer.one_cycle.cycle_max_mom,
                                               ctx.optimizer.one_cycle.decay_mom_rate,
                                               ctx.optimizer.one_cycle.last_batch_iteration)

    @torch.no_grad()
    def _to_device_detach(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.to(device=self.ctx.model.device, non_blocking=True).detach()

    def _forward_backward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(self.model(self._to_device_detach(src)), self._to_device_detach(tgt))
        loss.backward()
        return loss.detach()

    @torch.no_grad()
    def _clip_gradient(self):
        for p in self.gradients():
            g_norm = p.grad.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.zero_division_eps)
            p_norm = p.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.eps)
            grad_scale = (p_norm / g_norm * self.ctx.optimizer.agc.gradient_clipping).clamp(max=1)
            p.grad.data.copy_(p.grad * grad_scale)

    def accumulated_step(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        loss = sum(self._forward_backward(s.squeeze(0), t.squeeze(0)) for (s, t), _ in
                   zip(dataloader, range(self.ctx.optimizer.gradient_accumulation_steps)))
        self._clip_gradient()
        return loss

    @torch.no_grad()
    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

    @torch.no_grad()
    def gradients(self) -> torch.nn.Parameter:
        for p in self.model.parameters():
            if p.grad is None:
                continue
            yield p

    def save(self):
        torch.save(self.state_dict(), self.ctx.model.checkpoint_path)

    def load(self):
        wrong_keys = self.load_state_dict(torch.load(self.ctx.model.checkpoint_path), strict=False)
        for key in wrong_keys.missing_keys + wrong_keys.unexpected_keys:
            if not any(k.startswith('_') for k in key.split('.')):
                if key in wrong_keys.missing_keys:
                    raise ValueError(f"{key} is missing in checkpoint but exists in model")
                if key in wrong_keys.unexpected_keys:
                    raise ValueError(f"{key} is missing in model but exists in checkpoint")


class MomentumNetSide(torch.nn.Module):
    def __init__(self, beta: float):
        super(MomentumNetSide, self).__init__()
        self.beta = beta

    def forward(self, inp: torch.Tensor):
        return inp * self.beta


class LinearAttention(torch.nn.Module):
    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()
        self.embedding = torch.nn.Embedding(ctx.dataset.classes, ctx.model.features * 2).to(ctx.model.device)
        orthonormal(self.embedding.weight, ctx.model.input_embedding_std * 2 ** -0.5)

        pos_embd = torch.arange(0, ctx.model.sequence_length).unsqueeze(0) + 1
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float).to(ctx.model.device))

        cell = LinearAttentionCell(self, ctx, 1)
        self.stem = revlib.ReversibleSequential(*[c
                                                  for i in range(1, 1 + ctx.model.depth)
                                                  for c in [cell.momentum((1 - ctx.model.momentumnet_beta) /
                                                                          ctx.model.momentumnet_beta ** i),
                                                            MomentumNetSide(ctx.model.momentumnet_beta ** i)]],
                                               target_device=ctx.model.device)
        self.output = torch.nn.Conv1d(ctx.model.features * 2, ctx.dataset.classes, (1,)).to(ctx.model.device)
        torch.nn.init.zeros_(self.output.weight.data)

    def forward(self, inp: torch.Tensor):
        return self.output(self.stem(self.embedding(inp).transpose(1, 2)))

    def reset_cache(self):
        for mod in self.stem.modules():
            if isinstance(mod, LinearAttentionCell):
                mod.reset_cache()


class AuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        inp, = ctx.saved_tensors
        inp.mean().backward()


class ParameterStore(torch.nn.Module):
    """
    Something (likely deepspeed) changes all parameters in a ParameterList to [1] even though standalone parameters
    work. That's why a torch.nn.ModuleList of ParameterStores needs to be initialized.
    """

    def __init__(self, param: torch.Tensor):
        super(ParameterStore, self).__init__()
        self.param = torch.nn.Parameter(param)

    def __repr__(self):
        return (f'{self.__class__.__name__}(shape={str(list(self.param.size()))}, device={self.param.device}, '
                f'dtype={self.param.dtype})')


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.dropout_probability = 1 - ctx.model.dropout_probability
        self.bottleneck_group = ctx.model.bottleneck_group
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        experts = ctx.model.moe.num_experts
        moe_in_output = ctx.model.moe.use_in_output
        moe_in_input = ctx.model.moe.use_in_input
        param0 = ParameterStore(orthonormal([ctx.model.features, intermediate * 3], ctx.model.activation_std))
        param2 = ParameterStore(orthonormal([intermediate, ctx.model.features], 1))
        self.w0_gate = conv_weight(ctx.model.features, experts if moe_in_input else (3 * intermediate), 1, 1, 1)
        self.w0 = torch.nn.ModuleList([copy.deepcopy(param0) for _ in range(experts * moe_in_input)])
        self.w1 = conv_weight(intermediate, intermediate * 3, ctx.model.conv_kernel_size, ctx.model.bottleneck_group,
                              ctx.model.activation_std)
        self.w2_gate = conv_weight(intermediate, experts if moe_in_output else ctx.model.features, 1, 1, 1)
        self.w2 = torch.nn.ModuleList([copy.deepcopy(param2) for _ in range(experts * moe_in_output)])
        # Below is done to ignore pytorch's errors when calling .register_buffer without giving up the IDEs autocomplete
        self.idx: int = 0
        self._input_cache = torch.zeros([])
        self._cumsum_cache = torch.zeros([])

    def reset_cache(self):
        self._cumsum_cache = torch.zeros([])
        self._input_cache = torch.zeros([])
        self.idx = 0

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.training:
            div = self.divisor()
        elif self.caching:
            self.idx += inp.size(2)
            div = torch.LongTensor([self.idx]).to(inp.device)
        else:
            self.idx = inp.size(2)
            div = torch.arange(self.idx, device=inp.device).view(1, 1, -1) + 1
        loss0, loss1, self._input_cache, self._cumsum_cache, out = linear_attention(inp,
                                                                                    div,
                                                                                    self.w0_gate,
                                                                                    [store.param for store in self.w0],
                                                                                    self.w1,
                                                                                    self.w2_gate,
                                                                                    [store.param for store in self.w2],
                                                                                    self._input_cache,
                                                                                    self._cumsum_cache,
                                                                                    self.init_scale,
                                                                                    self.bottleneck_group,
                                                                                    self.dropout_probability,
                                                                                    self.training, self.caching,
                                                                                    self.idx)
        AuxLoss.apply(loss0 + loss1)
        return out

    def momentum(self, init_scale: float):
        out = copy.deepcopy(self)
        out.init_scale = init_scale
        return out
