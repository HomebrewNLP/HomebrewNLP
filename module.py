import torch


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def _single_calc(fn_input, sequence_input, linear_param, activate):
        features = fn_input.size(2)
        batch = fn_input.size(0)
        b = torch.bmm(fn_input, linear_param[0:1, :features].expand(batch, -1, -1))
        c = torch.bmm(sequence_input, linear_param[0:1, features:].expand(batch, -1, -1))
        a = torch.bmm(b,c)
        return torch.bmm(a,activate.expand(batch, -1, -1))

    @staticmethod
    def _calc(fn_input, sequence_input, linear_param, depth, activate):
        out = fn_input
        for idx in range(depth):
            out = ReversibleRNNFunction._single_calc(out, sequence_input, linear_param[idx:idx + 1], activate)
        return out

    @staticmethod
    def _forward_pass(fn_input, sequence_input, linear_param0, linear_param1, depth, activate):
        inp = fn_input.chunk(2, 1)
        outputs = [None, None]
        outputs[1] = inp[1] @ ReversibleRNNFunction._calc(inp[0], sequence_input, linear_param0, depth, activate)
        outputs[0] = inp[0] @ ReversibleRNNFunction._calc(outputs[1], sequence_input, linear_param1, depth, activate)
        out = torch.cat(outputs, 1)
        return out

    @staticmethod
    def _backward_one(out, inp, sequence_input, linear_param, depth, activate):
        tmp0 = ReversibleRNNFunction._calc(inp, sequence_input, linear_param, depth, activate)
        return out - tmp0

    @staticmethod
    def forward(ctx, fn_input, sequence_input, linear_param0, linear_param1, output_list, top, depth,
                activate, embedding):
        ctx.save_for_backward(sequence_input, linear_param0, linear_param1, activate, embedding)
        sequence_input = embedding[sequence_input]
        ctx.output_list = output_list
        ctx.top = top
        ctx.depth = depth

        if output_list:
            output_list.clear()
        with torch.no_grad():
            out = ReversibleRNNFunction._forward_pass(fn_input, sequence_input, linear_param0, linear_param1, depth,
                                                      activate)
        with torch.enable_grad():
            out.requires_grad_(True)
            output_list.append(out)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        sequence_input, linear_param0, linear_param1, activate, embedding = ctx.saved_tensors
        sequence_input = embedding[sequence_input]
        depth = ctx.depth
        if not sequence_input.requires_grad:
            return (None,) * 8
        out = ctx.output_list.pop(0)
        features = out.size(1) // 2
        out0, out1 = out[:, :features], out[:, features:]
        with torch.no_grad():
            inp0 = ReversibleRNNFunction._backward_one(out0, out1, sequence_input, linear_param1, depth, activate)
            inp1 = ReversibleRNNFunction._backward_one(out1, inp0, sequence_input, linear_param0, depth, activate)
        with torch.enable_grad():
            fn_input = torch.cat([inp0, inp1], 1)
            fn_input.detach_()
            fn_input.requires_grad_(True)
            args = (fn_input, sequence_input, linear_param0, linear_param1, depth, activate)
            grad_out = ReversibleRNNFunction._forward_pass(*args)
        grad_out.requires_grad_(True)
        grad_out = torch.autograd.grad(grad_out, (fn_input, sequence_input, linear_param0, linear_param1, activate),
                                       grad_output, allow_unused=True)
        fn_input.detach_()
        fn_input.requires_grad_(True)
        if not ctx.top:
            ctx.output_list.append(fn_input)
        return grad_out + (None,) * 4


class FixedRevRNN(torch.nn.Module):
    def __init__(self, input_cases, hidden_features, out_features, return_sequences=False, delay=8, depth=1,
                 input_count=0):
        super(FixedRevRNN, self).__init__()
        if input_count <= 0:
            raise UserWarning("No input count given")
        if hidden_features % 2:
            raise UserWarning(f"Ignoring uneven hidden feature and proceeding as if equal {hidden_features // 2 * 2}")

        self.return_sequences = return_sequences
        self.delay = delay
        self.hidden_features = hidden_features
        self.input_count = input_count

        hidden_features = hidden_features // 2

        self.hidden_state = torch.nn.Parameter(torch.zeros(2*hidden_features, hidden_features))
        self.hidden_state.data = self.hidden_state.data.unsqueeze(0)

        self.linear_param0 = torch.nn.Parameter(torch.zeros((depth,
                                                             2 * hidden_features,
                                                             hidden_features)))
        self.linear_param1 = torch.nn.Parameter(torch.zeros((depth,
                                                             2 * hidden_features,
                                                             hidden_features)))
        self.out_linear = torch.nn.Parameter(torch.randn((1, 2 * hidden_features, out_features)))
        self.register_buffer('activation', torch.ones((hidden_features, hidden_features)).qr()[0].unsqueeze(0))
        self.register_buffer('embedding', torch.ones((input_cases, hidden_features, hidden_features)))

        for idx in range(depth):
            for sub_idx in range(2):
                torch.nn.init.orthogonal_(self.linear_param0[idx][sub_idx*hidden_features:(1+sub_idx)*hidden_features])
                torch.nn.init.orthogonal_(self.linear_param1[idx][sub_idx*hidden_features:(1+sub_idx)*hidden_features])

        for idx in range(input_cases):
            torch.nn.init.orthogonal_(self.embedding[idx])

        torch.nn.init.orthogonal_(self.hidden_state[0][:hidden_features])
        torch.nn.init.orthogonal_(self.hidden_state[0][hidden_features:])


        self.depth = depth

    def forward(self, fn_input: torch.Tensor):
        # B, S, F, F
        output_list = []
        batch = fn_input.size(0)
        out = self.hidden_state.expand(batch, -1, -1)
        out.requires_grad_(True)
        input_features = self.hidden_features
        output = []
        top = True
        base_seq = seq = self.input_count
        seq += self.delay
        zeros = torch.eye(input_features, device=fn_input.device,
                          dtype=fn_input.dtype).unsqueeze(0).expand(batch, -1, -1)
        for idx in range(base_seq):
            out = ReversibleRNNFunction.apply(out, fn_input[:, idx], self.linear_param0, self.linear_param1,
                                              output_list, top, self.depth, self.activation, self.embedding)
            output.append(out)
            top = False
        for idx in range(base_seq, seq):
            out = ReversibleRNNFunction.apply(out, zeros, self.linear_param0, self.linear_param1,
                                              output_list, top, self.depth, self.activation, self.embedding)
            output.append(out)
            top = False
        out = torch.cat(output[self.delay:], 0)
        out = out.view(batch, base_seq, -1)
        out = torch.bmm(out, self.out_linear.expand(batch, -1, -1))
        return out


class Transpose(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input.transpose(1, 2)
