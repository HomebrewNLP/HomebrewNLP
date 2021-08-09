import typing

import torch


@torch.jit.script
def _activate_norm(fn_input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(fn_input)


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
    o = _activate_norm(b * c)
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


@torch.jit.script
def _fn_backward(out0: torch.Tensor, out1: torch.Tensor, sequence_input: torch.Tensor, linear_param0a: torch.Tensor,
                 linear_param0b: torch.Tensor, linear_param0c: torch.Tensor, linear_param1a: torch.Tensor,
                 linear_param1b: torch.Tensor, linear_param1c: torch.Tensor,
                 grad_output: typing.Tuple[torch.Tensor, torch.Tensor]
                 ):
    with torch.no_grad():
        inp0 = _backward_one(out0, out1, sequence_input, linear_param1a, linear_param1b, linear_param1c)
        inp1 = _backward_one(out1, inp0, sequence_input, linear_param0a, linear_param0b, linear_param0c)
    torch.is_grad_enabled()
    inp0.detach_()
    inp1.detach_()
    fn_input = _enable_grad((inp0, inp1))
    args = fn_input + (sequence_input, linear_param0a, linear_param0b, linear_param0c, linear_param1a,
                       linear_param1b, linear_param1c)
    grad_out = _enable_grad(_forward_pass(*args))
    grad_out = torch.autograd.grad(grad_out, args, grad_output, allow_unused=True)
    inp0.detach_()
    inp1.detach_()
    return grad_out, _enable_grad((inp0, inp1))


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor, sequence_input: torch.Tensor, linear_param0a: torch.Tensor,
                linear_param0b: torch.Tensor, linear_param0c: torch.Tensor, linear_param1a: torch.Tensor,
                linear_param1b: torch.Tensor, linear_param1c: torch.Tensor,
                output_list: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]],
                top: bool, embedding: torch.Tensor):
        ctx.save_for_backward(sequence_input, linear_param0a, linear_param0b, linear_param0c, linear_param1a,
                              linear_param1b, linear_param1c, embedding)
        sequence_input = embedding[sequence_input]
        ctx.output_list = output_list
        ctx.top = top

        if output_list:
            output_list.clear()

        out = _fn_forward(inp0, inp1, sequence_input, linear_param0a, linear_param0b, linear_param0c, linear_param1a,
                          linear_param1b, linear_param1c)
        output_list.append(out)
        return out

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor):
        sequence_input, l0a, l0b, l0c, l1a, l1b, l1c, embedding = ctx.saved_tensors
        sequence_input = embedding[sequence_input]
        if not sequence_input.requires_grad:
            return (None,) * 8

        out = ctx.output_list.pop(0)
        grad_out, fn_input = _fn_backward(out, sequence_input, l0a, l0b, l0c, l1a, l1b, l1c, (grad0, grad1))

        if not ctx.top:
            ctx.output_list.append(fn_input)
        return grad_out + (None,) * 3


fn = ReversibleRNNFunction().apply


class FixedRevRNN(torch.nn.Module):
    def __init__(self, input_cases, hidden_features, out_features, return_sequences=False, delay=8, input_count=0,
                 embedding_std=1):
        """

        :param input_cases: Input cases/max embedding index (not learned, can be extended)
        :param hidden_features: Base of a square feature matrix.
        :param out_features:
        :param return_sequences:
        :param delay:
        :param input_count:
        """
        super(FixedRevRNN, self).__init__()
        if input_count <= 0:
            raise UserWarning("No input count given")

        hidden_features = hidden_features ** 2
        self.return_sequences = return_sequences
        self.delay = delay
        self.input_count = input_count

        self.hidden_features = hidden_features

        features_sqrt = int(hidden_features ** 0.5)
        self.linear_param0a = torch.nn.Parameter(torch.randn((hidden_features, hidden_features)).qr().Q)
        self.linear_param0b = torch.nn.Parameter(torch.randn((hidden_features, hidden_features)).qr().Q)
        self.linear_param0c = torch.nn.Parameter(torch.randn((hidden_features, hidden_features)).qr().Q)
        self.linear_param1a = torch.nn.Parameter(torch.randn((hidden_features, hidden_features)).qr().Q)
        self.linear_param1b = torch.nn.Parameter(torch.randn((hidden_features, hidden_features)).qr().Q)
        self.linear_param1c = torch.nn.Parameter(torch.randn((hidden_features, hidden_features)).qr().Q)
        self.out_linear = torch.nn.Parameter(torch.randn((1, 2 * hidden_features, out_features)))
        self.embedding = torch.nn.Parameter(torch.randn((input_cases, hidden_features)).mul(embedding_std))

        self.register_buffer("hidden_state0", torch.randn(features_sqrt, features_sqrt).qr().Q.unsqueeze(0))
        self.register_buffer("hidden_state1", torch.randn(features_sqrt, features_sqrt).qr().Q.unsqueeze(0))

    def _call(self, out0: torch.Tensor, out1: torch.Tensor, sequence_input: torch.Tensor, top: bool,
              output: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]
              ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        out = fn(out0, out1, sequence_input, self.linear_param0a, self.linear_param0b, self.linear_param0c,
                 self.linear_param1a, self.linear_param1b, self.linear_param1c, output, top, self.embedding)
        output.append(out)
        return out

    def forward(self, fn_input: torch.Tensor):
        # B, S -> B, S, H, H -> B, S, F
        output_list = []
        batch = fn_input.size(0)
        out = (self.hidden_state0.expand(batch, -1, -1).requires_grad_(True),
               self.hidden_state0.expand(batch, -1, -1).requires_grad_(True))
        base_seq = seq = self.input_count
        seq += self.delay
        zeros = torch.zeros(1, device=fn_input.device, dtype=fn_input.dtype).expand(batch)
        for idx in range(base_seq):
            out = self._call(out[0], out[1], fn_input[:, idx], not idx, output_list)
        for idx in range(base_seq, seq):
            out = self._call(out[0], out[1], zeros, not idx, output_list)
        out = torch.stack(output_list[self.delay:], 1).view(batch, base_seq, -1)
        out = torch.bmm(out, self.out_linear.expand(batch, -1, -1))
        return out


class Transpose(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input.transpose(1, 2)
