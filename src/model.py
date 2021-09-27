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


class TripleNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale0: torch.Tensor, scale1: torch.Tensor, shift: torch.Tensor, norm_power: int):
        scale0_relu = scale0.relu()
        inp = scale0_relu.pow(3) * scale1 + shift
        inp = inp - inp.mean(1, True)
        rstd = inp.size(1) ** (1 / norm_power) / inp.norm(norm_power, 1, True)
        inp *= rstd
        if scale1.requires_grad:
            ctx.save_for_backward(scale0_relu, scale1, inp, rstd)
        return inp

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None, None, None
        scale0_relu, scale1, out, rstd = ctx.saved_tensors
        dout = dout * rstd
        dout -= (dout * out).mean(1, True) * out
        dout -= dout.mean(1, True)
        d_scale = dout * scale0_relu.square()
        return d_scale * scale1 * 3, d_scale * scale0_relu, dout, None


def conv(inp: torch.Tensor, weight: torch.Tensor, groups: int, use_pad: bool) -> torch.Tensor:
    if use_pad and weight.size()[-1] - 1 > 0:
        inp = torch.nn.functional.pad(inp, (weight.size()[-1] - 1, 0))
    return torch.nn.functional.conv1d(inp, weight, groups=groups)


def expert_matmul(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgf,gfo->bgo", inp, weight)


def moe(inp: torch.Tensor, expert_weights: torch.nn.ParameterList, feature_shuffle: torch.Tensor, groups: int,
        experts: int) -> torch.Tensor:
    batch, features, sequence = inp.size()
    permutation = torch.argsort(torch.rand(batch * sequence, device=inp.device)).long()
    permutation_inverse = torch.arange(batch * sequence, device=inp.device).gather(0, permutation)
    inp = inp.transpose(1, 2).reshape(batch * sequence, features)
    inp = inp.gather(0, permutation.view(-1, 1).expand_as(inp))
    if feature_shuffle is not None:
        inp = inp.gather(1, feature_shuffle.view(1, -1).expand_as(inp))
    inp = inp.view(batch * sequence // experts, experts * groups, features // groups)
    if len(expert_weights) == 1:
        inp = expert_matmul(inp, expert_weights[0])
    else:
        inp = torch.cat([expert_matmul(c, w) for c, w in zip(inp.chunk(len(expert_weights), 1), expert_weights)], -1)
    inp = inp.reshape(batch * sequence, -1)
    inp = inp.gather(0, permutation_inverse.view(-1, 1).expand_as(inp))
    inp = inp.view(batch, sequence, -1).transpose(1, 2)
    return inp


def moe_check(inp: torch.Tensor, w: torch.nn.ParameterList,
              feature_shuffle: torch.Tensor, groups: int, experts: int) -> torch.Tensor:
    return moe(inp, w, feature_shuffle, groups, experts) if experts > 0 else conv(inp, w[0], groups, False)


def linear_attention(inp: torch.Tensor, divisor: torch.Tensor,
                     w0: torch.nn.ParameterList,
                     feature_shuffle0: typing.Optional[torch.Tensor], groups0: int, experts0: int,
                     w1: torch.Tensor,
                     w2: torch.nn.ParameterList,
                     feature_shuffle2: typing.Optional[torch.Tensor], groups2: int, experts2: int,
                     input_cache: torch.Tensor, cumsum_cache: torch.Tensor, bottleneck_group: int, training: bool,
                     caching: bool, idx: int, norm_power: int
                     ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_size = w1.size(2)
    pad = True
    if not training and caching:
        if idx - 1 > kernel_size and inp.size(2) == 1:
            pad = False
            inp = torch.cat([input_cache, inp], -1)
        input_cache = inp[:, :, -kernel_size + 1:].detach()
    inp = moe_check(inp, w0, feature_shuffle0, groups0, experts0)
    depth, scale, shift = inp.chunk(3, 1)
    cum = depth.cumsum(-1)
    if not training and caching:
        cum = cum + cumsum_cache
        scale = scale[:, :, -1:]
        shift = shift[:, :, -1:]
        cum = cum[:, :, -1:]
        if idx - 1 > kernel_size:
            cumsum_cache = cum.detach()
    inp = TripleNorm.apply(cum / divisor, scale, shift, norm_power)
    inp = conv(inp, w1, bottleneck_group, pad)
    inp = TripleNorm.apply(*inp.chunk(3, 1), norm_power)
    inp = moe_check(inp, w2, feature_shuffle2, groups2, experts2)
    return input_cache, cumsum_cache, inp


def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv1d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)


class Trainer(torch.nn.Module):
    def __init__(self, ctx: Context, model: torch.nn.Module, data: typing.Optional[torch.Tensor]):
        super(Trainer, self).__init__()
        self.ctx = ctx
        self.model = torch.jit.trace(model, data) if data else model
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

    def accumulated_step(self, data: torch.Tensor) -> torch.Tensor:
        loss = sum(self._forward_backward(s, t) for s, t in zip(*data))
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


def get_moe_param(in_features: int, out_features: int, groups: int, experts: int, expert_chunks: int, std: float
                  ) -> typing.List[torch.nn.Parameter]:
    if experts:
        experts = groups if experts < 0 else experts
        out = orthonormal([in_features // groups, out_features // groups], std).view(1, in_features // groups, -1)
        out = out.expand(experts // expert_chunks * groups, -1, -1).detach().clone()
        return [torch.nn.Parameter(copy.deepcopy(out)) for _ in range(expert_chunks)]
    return [torch.nn.Parameter(conv_weight(in_features, out_features, 1, groups, std))]


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.bottleneck_group = ctx.model.bottleneck_group
        self.norm_power = ctx.model.norm_power
        self.groups0 = ctx.model.input_groups
        self.groups2 = ctx.model.output_groups
        self.experts0 = ctx.model.experts_in_input
        self.experts2 = ctx.model.experts_in_output
        self.expert_chunks = ctx.model.expert_chunks
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        self.w0 = torch.nn.ParameterList(get_moe_param(ctx.model.features, intermediate * 3, self.groups0,
                                                       self.experts0, self.expert_chunks, ctx.model.activation_std))
        self.w1 = conv_weight(intermediate, intermediate * 3, ctx.model.conv_kernel_size, ctx.model.bottleneck_group,
                              ctx.model.activation_std)
        self.w2 = torch.nn.ParameterList(get_moe_param(intermediate, ctx.model.features, self.groups2,
                                                       self.experts2, self.expert_chunks, 1))
        self.idx: int = 0
        self._input_cache = torch.zeros([])
        self._cumsum_cache = torch.zeros([])
        if ctx.model.feature_shuffle:
            self.register_buffer("feature_shuffle0", torch.argsort(torch.randn(ctx.model.features)).view(1, -1, 1))
            self.register_buffer("feature_shuffle2", torch.argsort(torch.randn(intermediate)).view(1, -1, 1))
        else:
            self.feature_shuffle0 = None
            self.feature_shuffle2 = None

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
        self._input_cache, self._cumsum_cache, out = linear_attention(inp, div,
                                                                      self.w0, self.feature_shuffle0, self.groups0,
                                                                      self.experts0,
                                                                      self.w1,
                                                                      self.w2, self.feature_shuffle2, self.groups2,
                                                                      self.experts2, self._input_cache,
                                                                      self._cumsum_cache, self.bottleneck_group,
                                                                      self.training, self.caching, self.idx,
                                                                      self.norm_power
                                                                      )
        out = out * self.init_scale
        return out

    def momentum(self, init_scale: float):
        out = copy.deepcopy(self)
        out.init_scale = init_scale
        return out
