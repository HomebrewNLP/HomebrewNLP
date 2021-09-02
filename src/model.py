import typing

import fmoe
import fmoe.gates
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


def get_moe(ctx: Context, in_features: int, out_features: int, gain: float) -> fmoe.FMoE:
    gate_args = {"d_model": ctx.model.features, "num_expert": ctx.model.moe.num_experts,
                 "world_size": 1, "top_k": ctx.model.moe.top_k}
    if ctx.model.moe.gate in ("GShardGate", "SwitchGate"):
        gate_args["topk"] = gate_args["top_k"]
        gate_args["capacity"] = (ctx.model.moe.capacity_factor, ctx.model.moe.eval_capacity_factor)
        del gate_args["top_k"]
    mod = fmoe.FMoE(ctx.model.moe.num_experts, in_features, top_k=ctx.model.moe.top_k)
    mod.experts = fmoe.FMoELinear(ctx.model.moe.num_experts, in_features, out_features, False)
    mod.experts.weight.data = orthonormal(torch.randn_like(mod.experts.weight), 1 / gain)
    mod.gate = getattr(fmoe.gates, ctx.model.moe.gate)(**gate_args)
    return mod


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(LinearAttentionCell, self).__init__()
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.groups = 3  # number of splits in ff
        self.dropout_probability = 1 - ctx.model.dropout_probability
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        self.inp = get_moe(ctx, ctx.model.features, intermediate, ctx.model.activation_std)
        self.w1 = conv_weight(intermediate, intermediate, ctx.model.conv_kernel_size, 1, ctx.model.activation_std)
        self.out = get_moe(ctx, intermediate, ctx.model.features * self.groups, 1)

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
        kernel_size = self.w1.size(2)
        if not self.training and self.caching:
            if self.idx > kernel_size:
                inp = torch.cat([self._input_cache, inp])
            self._input_cache = inp[:, :, -kernel_size:].detach()
            self.idx += 1
        print(inp.min(), inp.max())
        inp = self.inp(inp)
        inp = torch.relu(inp)
        inp = drop_conv(inp, self.w1, self.dropout_probability, self.training, 1)
        inp = torch.relu(inp)
        inp = self.out(inp)

        depth, scale, shift = inp.chunk(self.groups, 1)
        cum = depth.cumsum(1)
        if not self.training and self.caching:
            cum = cum + self._cumsum_cache
            self._cumsum_cache = cum[:, :, -kernel_size - 1].detach()
        return norm(cum / self.divisor() * scale + shift) * self.init_scale
