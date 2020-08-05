import torch


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def _single_calc(fn_input, sequence_input, linear_param):
        out = fn_input - fn_input.mean(dim=0, keepdim=True)
        features = out.size(1)
        out = torch.mm(out, linear_param[:features]) + torch.mm(sequence_input, linear_param[features:])
        return torch.nn.functional.relu6(out[:, :features]) * out[:, features:].tanh()

    @staticmethod
    def _calc(fn_input, sequence_input, linear_param, depth):
        out = fn_input
        for idx in range(depth):
            out = ReversibleRNNFunction._single_calc(out, sequence_input, linear_param[idx])
        return out

    @staticmethod
    def _forward_pass(fn_input, sequence_input, linear_param0, linear_param1, depth):
        inp = fn_input.chunk(2, 1)
        outputs = [None, None]
        outputs[1] = inp[1] + ReversibleRNNFunction._calc(inp[0], sequence_input, linear_param0, depth)
        outputs[0] = inp[0] + ReversibleRNNFunction._calc(outputs[1], sequence_input, linear_param1, depth)
        out = torch.cat(outputs, 1)
        return out

    @staticmethod
    def _backward_one(out, inp, sequence_input, linear_param, depth):
        tmp0 = ReversibleRNNFunction._calc(inp, sequence_input, linear_param, depth)
        return out - tmp0

    @staticmethod
    def forward(ctx, fn_input, _sequence_input, linear_param0, linear_param1, output_list, top, pos_enc, depth):
        ctx.save_for_backward(_sequence_input, linear_param0, linear_param1, pos_enc)
        ctx.output_list = output_list
        ctx.top = top
        ctx.depth = depth

        sequence_input = torch.cat([_sequence_input, pos_enc], -1)
        if output_list:
            output_list.clear()
        with torch.no_grad():
            out = ReversibleRNNFunction._forward_pass(fn_input, sequence_input, linear_param0, linear_param1, depth)
        with torch.enable_grad():
            out.requires_grad_(True)
            output_list.append(out)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        _sequence_input, linear_param0, linear_param1, pos_enc = ctx.saved_tensors
        depth = ctx.depth
        sequence_input = torch.cat([_sequence_input, pos_enc], -1)
        sequence_input.requires_grad_(_sequence_input.requires_grad)
        if not sequence_input.requires_grad:
            return (None,) * 8
        out = ctx.output_list.pop(0)
        features = out.size(1) // 2
        out0, out1 = out[:, :features], out[:, features:]
        with torch.no_grad():
            inp0 = ReversibleRNNFunction._backward_one(out0, out1, sequence_input, linear_param1, depth)
            inp1 = ReversibleRNNFunction._backward_one(out1, inp0, sequence_input, linear_param0, depth)
        with torch.enable_grad():
            fn_input = torch.cat([inp0, inp1], 1)
            fn_input.detach_()
            fn_input.requires_grad_(True)
            args = (fn_input, sequence_input, linear_param0, linear_param1, depth)
            grad_out = ReversibleRNNFunction._forward_pass(*args)
        grad_out.requires_grad_(True)
        grad_out = torch.autograd.grad(grad_out, (fn_input, _sequence_input, linear_param0, linear_param1), grad_output,
                                       allow_unused=True)
        fn_input.detach_()
        fn_input.requires_grad_(True)
        if not ctx.top:
            ctx.output_list.append(fn_input)
        return grad_out + (None,) * 4


class AdaptiveRevRNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, return_sequences=False, delay=8, depth=1):
        super(AdaptiveRevRNN, self).__init__()
        if hidden_features % 2:
            raise UserWarning(f"Ignoring uneven hidden feature and proceeding as if equal {hidden_features // 2 * 2}")

        self.return_sequences = return_sequences
        self.delay = delay
        self.input_features = input_features

        hidden_features = hidden_features // 2
        input_features += 2

        self.hidden_state = torch.nn.Parameter(torch.zeros(hidden_features * 2))
        torch.nn.init.normal_(self.hidden_state)

        self.linear_param0 = torch.nn.Parameter(torch.zeros((depth,
                                                             input_features + hidden_features,
                                                             2 * hidden_features)))
        self.linear_param1 = torch.nn.Parameter(torch.zeros((depth,
                                                             input_features + hidden_features,
                                                             2 * hidden_features)))

        for idx in range(depth):
            torch.nn.init.orthogonal_(self.linear_param0[idx])
            torch.nn.init.orthogonal_(self.linear_param1[idx])

        self.depth = depth

    def _apply_forward(self, itm, out, output_list, function_output, top, pos_enc):
        out = ReversibleRNNFunction.apply(out, itm, self.linear_param0, self.linear_param1, output_list, top, pos_enc,
                                          self.depth)
        if self.return_sequences:
            function_output.append(out)
        return out

    def forward(self, fn_input: torch.Tensor):
        output_list = []
        batch, seq, _ = fn_input.size()
        out = self.hidden_state.view(1, -1).expand(batch, -1)
        out.requires_grad_(fn_input.requires_grad)
        input_features = self.input_features
        output = []
        top = True
        base_seq = seq
        seq += self.delay
        zeros = torch.zeros((1, 1), device=fn_input.device, dtype=fn_input.dtype).expand(batch, input_features)
        factor = (seq + 1) / 2
        indices = torch.arange(1, seq + 1, dtype=fn_input.dtype, device=fn_input.device).view(1, -1, 1)
        positional_encoding = torch.cat([indices, (indices - factor) / factor], -1).expand(batch, -1, -1).to(
            fn_input.dtype)
        for idx in range(base_seq):
            out = self._apply_forward(fn_input[:, idx], out, output_list, output, top, positional_encoding[:, idx])
            top = False
        for idx in range(base_seq, seq):
            out = self._apply_forward(zeros, out, output_list, output, top, positional_encoding[:, idx])
            top = False
        if self.return_sequences:
            out = torch.stack(output[self.delay:], 1)
        return out


class FixedRevRNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, return_sequences=False, delay=8, depth=1, input_count=0):
        super(FixedRevRNN, self).__init__()
        if input_count <= 0:
            raise UserWarning("No input count given")
        if hidden_features % 2:
            raise UserWarning(f"Ignoring uneven hidden feature and proceeding as if equal {hidden_features // 2 * 2}")

        self.return_sequences = return_sequences
        self.delay = delay
        self.input_features = input_features
        self.input_count = input_count

        hidden_features = hidden_features // 2
        input_features += 2

        self.hidden_state = torch.nn.Parameter(torch.zeros(hidden_features * 2))
        torch.nn.init.normal_(self.hidden_state)

        self.linear_param0 = torch.nn.Parameter(torch.zeros((depth,
                                                             input_features + hidden_features,
                                                             2 * hidden_features)))
        self.linear_param1 = torch.nn.Parameter(torch.zeros((depth,
                                                             input_features + hidden_features,
                                                             2 * hidden_features)))

        for idx in range(depth):
            torch.nn.init.orthogonal_(self.linear_param0[idx])
            torch.nn.init.orthogonal_(self.linear_param1[idx])

        self.depth = depth

    def _apply_forward(self, itm, out, output_list, function_output, top, pos_enc):
        out = ReversibleRNNFunction.apply(out, itm, self.linear_param0, self.linear_param1, output_list, top, pos_enc,
                                          self.depth)
        function_output.append(out)
        return out

    def forward(self, fn_input: torch.Tensor):
        output_list = []
        batch, _, _ = fn_input.size()
        out = self.hidden_state.view(1, -1).expand(batch, -1)
        out.requires_grad_(fn_input.requires_grad)
        input_features = self.input_features
        output = []
        top = True
        base_seq = seq = self.input_count
        seq += self.delay
        zeros = torch.zeros((1, 1), device=fn_input.device, dtype=fn_input.dtype).expand(batch, input_features)
        factor = (seq + 1) / 2
        indices = torch.arange(1, seq + 1, dtype=fn_input.dtype, device=fn_input.device).view(1, -1, 1)
        positional_encoding = torch.cat([indices, (indices - factor) / factor], -1).expand(batch, -1, -1)
        for idx in range(base_seq):
            out = self._apply_forward(fn_input[:, idx], out, output_list, output, top, positional_encoding[:, idx])
            top = False
        for idx in range(base_seq, seq):
            out = self._apply_forward(zeros, out, output_list, output, top, positional_encoding[:, idx])
            top = False
        out = torch.cat(output[self.delay:], 0)[:, :self.input_features]
        out = out.view(batch, base_seq, -1)
        return out


class Transpose(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input.transpose(1, 2)
