import math
import typing

import revlib
import torch
import torch.nn.functional

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@torch.jit.script
def _activate_norm(fn_input: torch.Tensor) -> torch.Tensor:
    out = torch.nn.functional.relu(fn_input)
    out = out - out.mean(-1, keepdim=True)
    return out / ((out.square().sum(-1, keepdim=True).sqrt() + 1e-5) * out.size(-1) ** -0.5)


@torch.jit.script
def conv(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv1d(torch.nn.functional.pad(inp, (weight.size()[-1] - 1, 0)), weight)


@torch.jit.script
def feed_forward(inp: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    inp = conv(inp, w0)
    inp = _activate_norm(inp)
    inp = conv(inp, w1)
    inp = _activate_norm(inp)
    inp = conv(inp, w2)
    return inp


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_features: int, kernel_size: int, intermediate_factor: float):
        super().__init__()
        intermediate = int(hidden_features * intermediate_factor)
        self.w0 = torch.nn.Conv1d(hidden_features, intermediate, (1,), bias=False).weight
        self.w1 = torch.nn.Conv1d(intermediate, intermediate, (kernel_size,), bias=False).weight
        self.w2 = torch.nn.Conv1d(intermediate, hidden_features, (1,), bias=False).weight

    def forward(self, inp: torch.Tensor):
        return feed_forward(inp, self.w0, self.w1, self.w2)


@torch.jit.script
def linear_attention(inp: torch.Tensor, depth: torch.Tensor, point: torch.Tensor, shift: torch.Tensor,
                     divisor: torch.Tensor) -> torch.Tensor:
    return _activate_norm(inp * (depth.cumsum(1) / divisor + point) + shift)


class LinearAttention(torch.nn.Module):
    """
    One idea would be to run linear attention at every step in an rnn
    """

    def __init__(self, input_classes: int, hidden_features: int, out_features: int, depth: int = 8,
                 input_count: int = 0, embedding_std: float = 1, weight_shared_blocks: int = 1,
                 conv_kernel_size: int = 7, feed_forward_intermediate_factor: float = 2.):
        super(LinearAttention, self).__init__()
        self.embedding = torch.nn.Parameter(torch.randn((input_classes, hidden_features * 2)).mul(embedding_std))

        pos_embd = torch.arange(0, input_count).unsqueeze(0) + 1
        feature_embd = torch.arange(0, hidden_features).unsqueeze(1) + 1
        additive = (feature_embd % 2).to(torch.float)
        feature_embd = (feature_embd - additive) / 2
        additive *= math.pi
        feature_embd *= 8 / hidden_features
        feature_embd -= math.log(input_count / 2 / math.pi)
        feature_embd = torch.exp(feature_embd) + additive
        self.register_buffer("pos_embd", torch.sin(pos_embd * feature_embd).mul(embedding_std / depth).unsqueeze(0))
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float))
        self.stem = revlib.ReversibleSequential(*([LinearAttentionCell(hidden_features, self, conv_kernel_size,
                                                                       feed_forward_intermediate_factor)
                                                   for _ in range(depth)] * weight_shared_blocks))
        self.output = torch.nn.Conv1d(hidden_features * 2, out_features, 1)

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor):
        return torch.nn.functional.cross_entropy(self.output(self.stem(self.embedding[inp].transpose(1, 2))), tgt)


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, hidden_features: int, base: LinearAttention, kernel_size: int, intermediate_factor: float):
        super(LinearAttentionCell, self).__init__()
        self.pos_embd = lambda: base.pos_embd
        self.divisor = lambda: base.divisor
        self.depth = FeedForward(hidden_features, kernel_size, intermediate_factor)
        self.point = FeedForward(hidden_features, kernel_size, intermediate_factor)
        self.shift = FeedForward(hidden_features, kernel_size, intermediate_factor)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp + self.pos_embd()
        return linear_attention(inp, self.depth(out), self.point(out), self.shift(out), self.divisor())
