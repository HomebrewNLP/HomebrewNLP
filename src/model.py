import math
import typing

import deepspeed.moe.layer
import numpy as np
import revlib
import torch
import torch.nn.functional

from src.dataclass import Context

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def orthonormal(inp: typing.Union[torch.Tensor, torch.nn.Parameter], gain: float):
    original_input = inp
    if isinstance(inp, torch.nn.Parameter):
        inp = inp.data
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    a = torch.rand(flat_shape)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    inp.copy_((u if u.shape == flat_shape else v).reshape(inp.shape).mul(gain).to(device=inp.device, dtype=inp.dtype))
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


@torch.jit.script
def linear_attention(inp: torch.Tensor, divisor: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                     init_scale: float, cumsum_cache: torch.Tensor, dropout_probability: float, training: bool,
                     groups: int, caching: bool, input_cache: torch.Tensor,
                     idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_size = w1.size(2)
    if not training and caching:
        input_cache = inp[:, :, -kernel_size:]
        if idx - 1 > kernel_size:
            inp = torch.cat([input_cache, inp])
    inp = drop_conv(inp, w0, dropout_probability, training, 1)
    inp = torch.relu(inp)
    inp = drop_conv(inp, w1, dropout_probability, training, groups)
    inp = torch.relu(inp)
    inp = drop_conv(inp, w2, dropout_probability, training, groups)
    depth, scale, shift = inp.chunk(groups, 1)
    cum = depth.cumsum(1)
    if not training and caching:
        cum = cum + cumsum_cache
        cumsum_cache = cum[:, :, -kernel_size - 1].detach()
    return input_cache, cumsum_cache, norm(cum / divisor * scale + shift) * init_scale


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


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.groups = 3  # number of splits in ff
        self.dropout_probability = 1 - ctx.model.dropout_probability
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor) * self.groups
        self.w0 = conv_weight(ctx.model.features, intermediate, 1, 1, ctx.model.activation_std)
        self.w1 = conv_weight(intermediate, intermediate, ctx.model.conv_kernel_size, 3, ctx.model.activation_std)
        self.w2 = conv_weight(intermediate, ctx.model.features * self.groups, 1, 3, 1)
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
        self._input_cache, self._cumsum_cache, out = linear_attention(inp, self.divisor(), self.w0, self.w1, self.w2,
                                                                      self.init_scale, self._cumsum_cache,
                                                                      self.dropout_probability, self.training,
                                                                      self.groups, self.caching,
                                                                      self._input_cache, self.idx)
        self.idx += 1
        return out
