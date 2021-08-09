import typing

import torch
import torch.nn.functional


@torch.jit.script
def _activate_norm(fn_input: torch.Tensor) -> torch.Tensor:
    out = torch.nn.functional.relu(fn_input)
    out = out - out.mean(-1, keepdim=True)
    return out / (torch.sqrt(torch.square(out).sum(-1, keepdim=True) + 1e-5) * out.size(-1) ** 0.5)


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


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor, tmp_inp0: torch.Tensor, tmp_inp1: torch.Tensor):
        ctx.save_for_backward(tmp_inp0, tmp_inp1)
        return inp0, inp1

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor):
        tmp_inp0, tmp_inp1 = ctx.saved_tensors
        return grad0, grad1, tmp_inp0, tmp_inp1


replace = ReplaceGrad().apply
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
        self.out_linear = torch.nn.Parameter(torch.zeros((1, 2 * hidden_features, out_features)))
        self.embedding = torch.nn.Parameter(torch.randn((input_cases, hidden_features)).mul(embedding_std))

        self.register_buffer("hidden_state0", torch.randn(features_sqrt, features_sqrt).qr().Q.unsqueeze(0))
        self.register_buffer("hidden_state1", torch.randn(features_sqrt, features_sqrt).qr().Q.unsqueeze(0))

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
        out = replace(*out)
        outputs = outputs[:-2] + list(out)
        out = torch.cat(outputs[self.delay:], 1).view(batch, base_seq, -1)
        return torch.bmm(out, self.out_linear.expand(batch, -1, -1))


class Transpose(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input.transpose(1, 2)
