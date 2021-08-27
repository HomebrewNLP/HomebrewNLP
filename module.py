import math
import typing

import torch
import torch.nn.functional

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@torch.jit.script
def _activate_norm(fn_input: torch.Tensor) -> torch.Tensor:
    out = torch.nn.functional.relu(fn_input)
    out = out - out.mean(-1, keepdim=True)
    return out / ((out.square().sum(-1, keepdim=True).sqrt() + 1e-5) * out.size(-1) ** -0.5)


@torch.jit.script
def _calc(fn_input: torch.Tensor, sequence_input: torch.Tensor, linear_param_a: torch.Tensor,
          linear_param_b: torch.Tensor, linear_param_c: torch.Tensor) -> torch.Tensor:
    features_sqrt = int(fn_input.size(2))
    batch = int(fn_input.size(0))
    features = int(features_sqrt ** 2)
    fn_input = fn_input.reshape(batch, features)
    fn_input = _activate_norm(fn_input)
    b = torch.mm(fn_input, linear_param_a)
    c = torch.mm(sequence_input, linear_param_b)
    o = _activate_norm(b + c)
    o = torch.mm(o, linear_param_c)
    o = o.reshape(batch, features_sqrt, features_sqrt)
    return o.qr().Q


@torch.jit.script
def _forward_pass(inp0: torch.Tensor, inp1: torch.Tensor, sequence_input: torch.Tensor, linear_param0a: torch.Tensor,
                  linear_param0b: torch.Tensor, linear_param0c: torch.Tensor, linear_param1a: torch.Tensor,
                  linear_param1b: torch.Tensor,
                  linear_param1c: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    out1 = torch.bmm(inp1, _calc(inp0, sequence_input, linear_param0a, linear_param0b, linear_param0c))
    out0 = torch.bmm(inp0, _calc(out1, sequence_input, linear_param1a, linear_param1b, linear_param1c))
    return out0, out1


@torch.jit.script
def _backward_one(out: torch.Tensor, inp: torch.Tensor, sequence_input: torch.Tensor,
                  linear_param_a: torch.Tensor, linear_param_b: torch.Tensor,
                  linear_param_c: torch.Tensor) -> torch.Tensor:
    return torch.bmm(out, _calc(inp, sequence_input, linear_param_a, linear_param_b, linear_param_c).transpose(1, 2))


@torch.jit.script
def _enable_grad(inp: typing.Tuple[torch.Tensor, torch.Tensor]) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    x0, x1 = inp
    torch.is_grad_enabled()
    x0.requires_grad_(True)
    x1.requires_grad_(True)
    return x0, x1


@torch.jit.script
def _fn_forward(inp0: torch.Tensor, inp1: torch.Tensor, sequence_input: torch.Tensor, linear_param0a: torch.Tensor,
                linear_param0b: torch.Tensor, linear_param0c: torch.Tensor, linear_param1a: torch.Tensor,
                linear_param1b: torch.Tensor, linear_param1c: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = _forward_pass(inp0, inp1, sequence_input, linear_param0a, linear_param0b, linear_param0c, linear_param1a,
                            linear_param1b, linear_param1c)
    return _enable_grad(out)


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor,
                linear_param0a: torch.Tensor, linear_param0b: torch.Tensor, linear_param0c: torch.Tensor,
                linear_param1a: torch.Tensor, linear_param1b: torch.Tensor, linear_param1c: torch.Tensor,
                embedding: torch.Tensor, tmp_inp0: torch.Tensor, tmp_inp1: torch.Tensor, sequence_input: torch.Tensor,
                bottom: bool):
        ctx.save_for_backward(sequence_input, linear_param0a, linear_param0b, linear_param0c, linear_param1a,
                              linear_param1b, linear_param1c, embedding)
        with torch.no_grad():
            sequence_input = embedding[sequence_input].detach()

            out = _fn_forward(inp0, inp1, sequence_input, linear_param0a, linear_param0b, linear_param0c,
                              linear_param1a,
                              linear_param1b, linear_param1c)
            return out + (inp0, inp1)

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor):
        sequence_input, l0a, l0b, l0c, l1a, l1b, l1c, embedding = ctx.saved_tensors
        sequence_input = embedding[sequence_input]
        with torch.no_grad():
            inp0 = _backward_one(out0, out1, sequence_input, l1a, l1b, l1c)
            inp1 = _backward_one(out1, inp0, sequence_input, l0a, l0b, l0c)
        with torch.enable_grad():
            inp0.detach_()
            inp1.detach_()
            fn_input = _enable_grad((inp0, inp1))
            args = (sequence_input, l0a, l0b, l0c, l1a, l1b, l1c)
            grad_out = _enable_grad(_forward_pass(*fn_input, *args))
            grad_out = torch.autograd.grad(grad_out, fn_input + args[1:] + (embedding,), (grad0, grad1),
                                           allow_unused=True)
        return grad_out + fn_input + (None, None,)


class ReplaceGrad2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor, tmp_inp0: torch.Tensor, tmp_inp1: torch.Tensor):
        ctx.save_for_backward(tmp_inp0, tmp_inp1)
        return inp0, inp1

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor):
        tmp_inp0, tmp_inp1 = ctx.saved_tensors
        return grad0, grad1, tmp_inp0, tmp_inp1


class ReplaceGrad1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor, tmp_inp0: torch.Tensor, tmp_inp1: torch.Tensor):
        ctx.save_for_backward(tmp_inp0)
        return inp0, inp1

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor):
        tmp_inp0, = ctx.saved_tensors
        return grad0, tmp_inp0, grad1, torch.zeros_like(tmp_inp0)


class ReversibleHalfResidualSwapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0: torch.Tensor, back_x0: torch.Tensor, x1: torch.Tensor, back_x1: torch.Tensor,
                mod: torch.nn.Module) -> QUAD_TENSOR:
        ctx.mod = mod
        return x1, back_x0, x0 + mod(x1), back_x1

    @staticmethod
    def backward(ctx, dy0: torch.Tensor, y0: torch.Tensor, dy1: torch.Tensor, y1: torch.Tensor
                 ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        with torch.enable_grad():
            y0 = y0.requires_grad_(True)
            out = ctx.mod(y0)
        with torch.no_grad():
            x0 = y1 - out.detach()
        with torch.enable_grad():
            dx0, *param_grad = torch.autograd.grad(out, (y0,) + tuple(ctx.mod.parameters()), dy1)
        with torch.no_grad():
            for p, g in zip(ctx.mod.parameters(), param_grad):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.data.add_(g)
        with torch.enable_grad():
            return dy1.detach(), x0.detach(), dx0.add(dy0).detach(), y0.detach(), None


replace1 = ReplaceGrad1().apply
replace2 = ReplaceGrad2().apply
fn = ReversibleRNNFunction().apply
reverse = ReversibleHalfResidualSwapFn().apply


class ReversibleModule(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module):
        super(ReversibleModule, self).__init__()
        self.wrapped_module = wrapped_module

    def forward(self, inp: QUAD_TENSOR) -> QUAD_TENSOR:
        return reverse(*inp, self.wrapped_module)


class ReversibleSequential(torch.nn.Module):
    def __init__(self, *modules, split_dim=1):
        super(ReversibleSequential, self).__init__()
        self.stem = torch.nn.Sequential(*[m if isinstance(m, ReversibleModule) else ReversibleModule(m)
                                          for m in modules])
        self.split_dim = split_dim

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp0, inp1 = inp.chunk(2, self.split_dim)
        zeros = torch.zeros_like(inp0)
        return torch.cat(replace1(*self.stem((inp0, zeros, inp1, zeros))), dim=self.split_dim)


def orthogonal(size: int) -> torch.Tensor:
    return torch.randn((size, size)).qr().Q


def orthogonal_param(size: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(orthogonal(size))


def orthogonal_param_batch(size: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(orthogonal(size).unsqueeze(0))


def output(hidden_features, out_features):
    return torch.nn.Conv1d(hidden_features, out_features, 1)


class FixedRevRNN(torch.nn.Module):
    def __init__(self, input_cases, hidden_features, out_features, delay=8, input_count=0, embedding_std=1):
        """

        :param input_cases: Input cases/max embedding index (not learned, can be extended)
        :param hidden_features: Base of a square feature matrix.
        :param out_features:
        :param delay:
        :param input_count:
        """
        super(FixedRevRNN, self).__init__()
        if input_count <= 0:
            raise UserWarning("No input count given")

        hidden_features = hidden_features ** 2
        self.delay = delay
        self.input_count = input_count

        self.hidden_features = hidden_features

        features_sqrt = int(hidden_features ** 0.5)
        self.linear_param0a = orthogonal_param(hidden_features)
        self.linear_param0b = orthogonal_param(hidden_features)
        self.linear_param0c = orthogonal_param(hidden_features)
        self.linear_param1a = orthogonal_param(hidden_features)
        self.linear_param1b = orthogonal_param(hidden_features)
        self.linear_param1c = orthogonal_param(hidden_features)
        self.embedding = torch.nn.Parameter(torch.randn((input_cases, hidden_features)).mul(embedding_std))
        self.output = output(hidden_features * 2, out_features)

        self.register_buffer("hidden_state0", orthogonal(features_sqrt).unsqueeze(0))
        self.register_buffer("hidden_state1", orthogonal(features_sqrt).unsqueeze(0))

    def _call(self, inp: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              sequence_inp: torch.Tensor,
              bottom: bool) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return fn(inp[0], inp[1], self.linear_param0a, self.linear_param0b, self.linear_param0c, self.linear_param1a,
                  self.linear_param1b, self.linear_param1c, self.embedding, inp[2], inp[3], sequence_inp, bottom)

    def forward(self, fn_input: torch.Tensor):
        # B, S -> B, S, H, H -> B, S, F
        batch = fn_input.size(0)
        out = (self.hidden_state0.expand(batch, -1, -1).requires_grad_(True),
               self.hidden_state1.expand(batch, -1, -1).requires_grad_(True)) * 2
        base_seq = seq = self.input_count
        seq += self.delay
        outputs: typing.List[typing.Optional[torch.Tensor]] = []
        zeros = torch.zeros(1, device=fn_input.device, dtype=fn_input.dtype).expand(batch)
        for idx in range(base_seq):
            out = self._call(out, fn_input[:, idx], not idx)
            if idx >= self.delay:
                outputs.extend(list(out)[:2])
        for idx in range(base_seq, seq):
            out = self._call(out, zeros, not idx)
            outputs.extend(list(out)[:2])
        out = replace2(*out)
        outputs = outputs[:-2] + list(out)
        out = torch.cat(outputs[self.delay:], 1).view(batch, base_seq, -1).transpose(1, 2)
        return self.output(out)


class Transpose(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input.transpose(1, 2)


@torch.jit.script
def conv(inp: torch.Tensor, weight: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return torch.nn.functional.conv1d(torch.nn.functional.pad(inp, (kernel_size - 1, 0)), weight)


@torch.jit.script
def feed_forward(inp: torch.Tensor, weight0: torch.Tensor, weight1: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return conv(_activate_norm(conv(inp, weight0, kernel_size)), weight1, kernel_size)


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_features, kernel_size=7, intermediate_factor=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.w0 = torch.nn.Conv1d(hidden_features, hidden_features * intermediate_factor, kernel_size,
                                  bias=False).weight
        self.w1 = torch.nn.Conv1d(hidden_features * intermediate_factor, hidden_features, kernel_size,
                                  bias=False).weight

    def forward(self, inp: torch.Tensor):
        return feed_forward(inp, self.w0, self.w1, self.kernel_size)


@torch.jit.script
def linear_attention(inp: torch.Tensor, depth: torch.Tensor, point: torch.Tensor, shift: torch.Tensor,
                     divisor: torch.Tensor) -> torch.Tensor:
    return _activate_norm(inp * (depth.cumsum(1) / divisor + point) + shift)


class LinearAttentionCell(torch.nn.Module):
    def __init__(self, hidden_features, base):
        super(LinearAttentionCell, self).__init__()
        self.pos_embd = lambda: base.pos_embd
        self.divisor = lambda: base.divisor
        self.depth = FeedForward(hidden_features)
        self.point = FeedForward(hidden_features)
        self.shift = FeedForward(hidden_features)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp + self.pos_embd()
        return linear_attention(inp, self.depth(out), self.point(out), self.shift(out), self.divisor())


class LinearAttention(torch.nn.Module):
    """
    One idea would be to run linear attention at every step in an rnn
    """

    def __init__(self, input_cases, hidden_features, out_features, delay=8, input_count=0, embedding_std=1):
        super(LinearAttention, self).__init__()

        hidden_features = hidden_features ** 2

        self.embedding = torch.nn.Parameter(torch.randn((input_cases, hidden_features * 2)).mul(embedding_std))

        pos_embd = torch.arange(0, input_count).unsqueeze(0) + 1
        feature_embd = torch.arange(0, hidden_features).unsqueeze(1) + 1
        additive = (feature_embd % 2).to(torch.float)
        feature_embd = (feature_embd - additive) / 2
        additive *= math.pi
        feature_embd *= 8 / hidden_features
        feature_embd -= math.log(input_count / 2 / math.pi)
        feature_embd = torch.exp(feature_embd) + additive
        self.register_buffer("pos_embd", torch.sin(pos_embd * feature_embd).mul(embedding_std / delay).unsqueeze(0))
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float))
        self.stem = ReversibleSequential(*[LinearAttentionCell(hidden_features, self) for _ in range(delay)])
        self.output = output(hidden_features * 2, out_features)

    def forward(self, inp: torch.Tensor, tgt: torch.Tensor):
        return torch.nn.functional.cross_entropy(self.output(self.stem(self.embedding[inp].transpose(1, 2))), tgt)
