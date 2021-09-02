import typing

import numpy as np
import revlib
import torch
import torch.nn.functional

from src.dataclass import Context

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


def conv(inp: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    pad = weight.size()[-1] - 1
    if pad:
        inp = torch.nn.functional.pad(inp, (pad, 0))
    return torch.nn.functional.conv1d(inp, weight, groups=groups)


def drop_conv(inp: torch.Tensor, weight: torch.Tensor, p: float, train: bool, groups: int) -> torch.Tensor:
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
    return conv(inp, weight, groups)


def moe(inp: torch.Tensor, w: typing.List[torch.Tensor],
        gate: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    out = torch.nn.functional.conv1d(inp, gate)
    gates = torch.nn.functional.softmax(out, dim=1)
    one_hot = torch.nn.functional.one_hot(torch.argmax(out, dim=1), num_classes=out.shape[1]).to(dtype=torch.bool)
    inp_t = inp.transpose(1, 2)
    batch, features, sequence = inp.size()
    out = torch.empty((batch * sequence, w[0].size(1)), device=inp.device, dtype=inp.dtype)
    for expert, param in zip(one_hot.unbind(-1), w):
        tmp = torch.masked_select(inp_t, expert.unsqueeze(2)).view(-1, features).mm(param)
        out = out.masked_scatter(expert.view(-1, 1), tmp)
    loss = torch.sum(torch.mean(gates, dim=(0, 2)) * torch.mean(one_hot.float(), dim=(0, 1)))
    return loss, out.view(batch, sequence, -1).transpose(1, 2)


def moe_check(inp: torch.Tensor, w_gate: torch.Tensor, w: typing.List[torch.Tensor], dropout_probability: float,
              training: bool, groups: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    if w:
        return moe(inp, w, w_gate)
    return (torch.zeros([1], device=inp.device, dtype=inp.dtype),
            drop_conv(inp, w_gate, dropout_probability, training, groups))


@torch.jit.script
def linear_attention(inp: torch.Tensor, divisor: torch.Tensor, w0_gate: torch.Tensor,
                     w0: typing.List[torch.Tensor], w1: torch.Tensor, w2_gate: torch.Tensor,
                     w2: typing.List[torch.Tensor], input_cache: torch.Tensor, cumsum_cache: torch.Tensor,
                     init_scale: float, bottleneck_group: int, dropout_probability: float, training: bool, groups: int,
                     caching: bool, idx: int
                     ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_size = w1.size(2)
    if not training and caching:
        input_cache = inp[:, :, -kernel_size:].detach()
        if idx - 1 > kernel_size:
            inp = torch.cat([input_cache, inp])
    loss0, inp = moe_check(inp, w0_gate, w0, dropout_probability, training, 1)
    inp = torch.relu(inp)
    inp = drop_conv(inp, w1, dropout_probability, training, bottleneck_group)
    inp = torch.relu(inp)
    loss1, inp = moe_check(inp, w2_gate, w2, dropout_probability, training, 1)
    depth, scale, shift = inp.chunk(groups, 1)
    cum = depth.cumsum(1)
    if not training and caching:
        cum = cum + cumsum_cache
        cumsum_cache = cum[:, :, -kernel_size - 1].detach()
    return loss0, loss1, input_cache, cumsum_cache, norm(cum / divisor * scale + shift) * init_scale


def get_coupling(beta_tmp: float):
    @torch.jit.script
    def momentum_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor, beta: float) -> torch.Tensor:
        return other_stream * beta + fn_out * (1 - beta)

    @torch.jit.script
    def momentum_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor, beta: float) -> torch.Tensor:
        return (output - fn_out * (1 - beta)) / beta

    def _wrapped_momentum_coupling_forward(x, y):
        return momentum_coupling_forward(x, y, beta_tmp)

    def _wrapped_momentum_coupling_inverse(x, y):
        return momentum_coupling_inverse(x, y, beta_tmp)

    return _wrapped_momentum_coupling_forward, _wrapped_momentum_coupling_inverse


def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv1d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)


class LinearAttention(torch.nn.Module):
    """
    One idea would be to run linear attention at every step in an rnn
    """

    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()

        self.embedding = torch.nn.Embedding(ctx.dataset.classes, ctx.model.features * 2)
        self.embedding.weight.data.mul_(ctx.model.input_embedding_std * 2 ** -0.5)

        init_scale = ctx.model.depth ** -0.5
        pos_embd = torch.arange(0, ctx.model.sequence_length).unsqueeze(0) + 1
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float))

        momentum_coupling_forward, momentum_coupling_inverse = get_coupling(ctx.model.momentumnet_beta)
        self.stem = revlib.ReversibleSequential(*([layer
                                                   for _ in range(ctx.model.depth)
                                                   for layer in [LinearAttentionCell(self, ctx, init_scale),
                                                                 torch.nn.Identity()]]),
                                                coupling_forward=[momentum_coupling_forward,
                                                                  revlib.additive_coupling_forward],
                                                coupling_inverse=[momentum_coupling_inverse,
                                                                  revlib.additive_coupling_inverse])
        self.output = torch.nn.Conv1d(ctx.model.features * 2, ctx.dataset.classes, (1,))
        torch.nn.init.zeros_(self.output.weight.data)

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor):
        out = self.output(self.stem(self.embedding(inp).transpose(1, 2)))
        if not self.training:
            return out
        return torch.nn.functional.cross_entropy(out, tgt)


class AuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        inp, = ctx.saved_tensors
        inp.mean().backward()
        return None


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
        self.groups = 3  # number of splits in ff
        self.dropout_probability = 1 - ctx.model.dropout_probability
        self.bottleneck_group = ctx.model.bottleneck_group
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        experts = ctx.model.moe.num_experts
        moe_in_output = ctx.model.moe.use_in_output
        moe_in_input = ctx.model.moe.use_in_input
        self.w0_gate = conv_weight(ctx.model.features, experts if moe_in_input else intermediate, 1, 1, 1)
        self.w0 = torch.nn.ModuleList([ParameterStore(orthonormal([ctx.model.features, intermediate],
                                                                  ctx.model.activation_std))
                                       for _ in range(experts * moe_in_input)])
        self.w1 = conv_weight(intermediate, intermediate, ctx.model.conv_kernel_size, ctx.model.bottleneck_group,
                              ctx.model.activation_std)
        self.w2_gate = conv_weight(intermediate, experts if moe_in_output else (ctx.model.features * self.groups), 1,
                                   1, 1)
        self.w2 = torch.nn.ModuleList([ParameterStore(orthonormal([intermediate, self.groups * ctx.model.features], 1))
                                       for _ in range(experts * moe_in_output)])
        # Below is done to ignore pytorch's errors when calling .register_buffer without giving up the IDEs autocomplete
        self.idx: int = 0
        self._input_cache = torch.zeros([])
        self._cumsum_cache = torch.zeros([])
        self._buffers["_cumsum_cache"] = self._cumsum_cache
        self._buffers["_input_cache"] = self._input_cache
        self._non_persistent_buffers_set.discard("_cumsum_cache")
        self._non_persistent_buffers_set.discard("_input_cache")

    def reset_cache(self):
        self._cumsum_cache.mul_(0)
        self._input_cache.mul_(0)
        self.idx = 0

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        loss0, loss1, self._input_cache, self._cumsum_cache, out = linear_attention(inp, self.divisor(), self.w0_gate,
                                                                                    [store.param for store in self.w0],
                                                                                    self.w1, self.w2_gate,
                                                                                    [store.param for store in self.w2],
                                                                                    self._input_cache,
                                                                                    self._cumsum_cache, self.init_scale,
                                                                                    self.bottleneck_group,
                                                                                    self.dropout_probability,
                                                                                    self.training, self.groups,
                                                                                    self.caching, self.idx)
        AuxLoss.apply(loss0 + loss1)
        self.idx += 1
        return out
