import torch


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    return fn_input * torch.tanh(torch.nn.functional.softplus(fn_input))


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def _calc(fn_input, sequence_input, bn_weight, bn_bias, linear_param):
        std, mean = torch.std_mean(fn_input, 0, keepdim=True)
        out = (fn_input - mean) / (std + 1e-6) * bn_weight + bn_bias
        out = mish(out)
        out0, out1 = torch.nn.functional.linear(torch.cat([out, sequence_input], 1), linear_param, None).chunk(2, 1)
        out1 = out1.sigmoid()
        return out0 * out1, 1 - out1

    @staticmethod
    def _forward_pass(fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1,
                      bn_bias1):
        inp = fn_input.chunk(2, 1)
        print(inp[0].mean().item(), inp[1].mean().item(), "F")

        out = inp[0]
        params = [[bn_weight0, bn_bias0, linear_param0], [bn_weight1, bn_bias1, linear_param1]]
        outputs = [None, None]
        for i in range(2):
            tmp0, tmp1 = ReversibleRNNFunction._calc(out, sequence_input, *params[i])
            outputs[1 - i] = out = tmp0 + inp[1 - i] * tmp1
        out = torch.cat(outputs, 1)
        return out

    @staticmethod
    def _backward_one(out, inp, sequence_input, bn_weight, bn_bias, linear_param):
        tmp0, tmp1 = ReversibleRNNFunction._calc(inp, sequence_input, bn_weight, bn_bias, linear_param)
        return (out - tmp0) / tmp1

    @staticmethod
    def forward(ctx, fn_input, _sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1,
                bn_bias1, top, pos_enc):
        output_list = []
        ctx.output_list = output_list
        ctx.top = top

        sequence_input = torch.cat([_sequence_input, pos_enc], -1)
        if output_list:
            output_list.clear()
        with torch.no_grad():
            print("FORWARD")
            out = ReversibleRNNFunction._forward_pass(fn_input, sequence_input, linear_param0, linear_param1,
                                                      bn_weight0, bn_bias0, bn_weight1, bn_bias1)
        with torch.enable_grad():
            out.requires_grad_(True)
            ctx.save_for_backward(_sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1,
                                  bn_bias1,
                                  pos_enc, out)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        print("BACK")
        sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1, pos_enc, out = ctx.saved_tensors
        _sequence_input = torch.cat([sequence_input, pos_enc], -1)
        _sequence_input.requires_grad_(sequence_input.requires_grad)
        sequence_input = _sequence_input
        if not sequence_input.requires_grad:
            print("NO GRAD")
            return (None,) * 10
        out0, out1 = out.chunk(2, 1)
        with torch.no_grad():
            inp0 = ReversibleRNNFunction._backward_one(out0, out1, sequence_input, bn_weight1, bn_bias1, linear_param1)
            inp1 = ReversibleRNNFunction._backward_one(out1, inp0, sequence_input, bn_weight0, bn_bias0, linear_param0)
        print(inp0.mean().item(), inp1.mean().item(), "B")
        with torch.enable_grad():
            fn_input = torch.cat([inp0, inp1], 1)
            fn_input.detach_()
            fn_input.requires_grad_(True)
            args = (fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1)
            grad_out = ReversibleRNNFunction._forward_pass(*args)
        grad_out.requires_grad_(True)
        grad_out = list(torch.autograd.grad(grad_out, args, grad_output, allow_unused=True))
        grad_out[1] = grad_out[1][:, :-2]
        fn_input.detach_()
        fn_input.requires_grad_(True)
        if not ctx.top:
            ctx.output_list.append(fn_input)
        print("OUT")
        return tuple(grad_out) + (None, None)


class RevRNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, return_sequences=False, delay=8, depth=1):
        super(RevRNN, self).__init__()
        if hidden_features % 2:
            raise UserWarning(f"Ignoring uneven hidden feature and proceeding as if equal {hidden_features // 2 * 2}")

        self.return_sequences = return_sequences
        self.delay = delay
        self.input_features = input_features

        hidden_features = hidden_features // 2
        input_features += 2

        self.hidden_state = torch.nn.Parameter(torch.zeros(hidden_features * 2))
        torch.nn.init.normal_(self.hidden_state)

        def set_param(self, idx):
            linear_param0 = torch.nn.Parameter(torch.zeros((2 * hidden_features, input_features + hidden_features)))
            linear_param1 = torch.nn.Parameter(torch.zeros((2 * hidden_features, input_features + hidden_features)))
            bn_weight0 = torch.nn.Parameter(torch.ones(hidden_features))
            bn_weight1 = torch.nn.Parameter(torch.ones(hidden_features))
            bn_bias0 = torch.nn.Parameter(torch.zeros(hidden_features))
            bn_bias1 = torch.nn.Parameter(torch.zeros(hidden_features))

            torch.nn.init.orthogonal_(linear_param0)
            torch.nn.init.orthogonal_(linear_param1)
            torch.nn.init.uniform_(bn_weight0, 0.998, 1.002)
            torch.nn.init.uniform_(bn_weight1, 0.998, 1.002)
            torch.nn.init.uniform_(bn_bias0, -0.002, 0.002)
            torch.nn.init.uniform_(bn_bias1, -0.002, 0.002)

            setattr(self, f'linear_param0{idx}', linear_param0)
            setattr(self, f'linear_param1{idx}', linear_param1)
            setattr(self, f'bn_weight0{idx}', bn_weight0)
            setattr(self, f'bn_weight1{idx}', bn_weight1)
            setattr(self, f'bn_bias0{idx}', bn_bias0)
            setattr(self, f'bn_bias1{idx}', bn_bias1)

            return [linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1]

        self.parameter_list = [set_param(self, i) for i in range(depth)]

    def _apply_forward(self, itm, out, output_list, function_output, top, pos_enc):
        for param in self.parameter_list:
            out = ReversibleRNNFunction.apply(out, itm, *param, output_list, top, pos_enc)
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
        factor = (seq + 1) / 2
        indices = torch.arange(1, seq + 1, dtype=torch.float, device=fn_input.device).view(1, -1, 1)
        positional_encoding = torch.cat([indices, (indices - factor) / factor], -1).expand(batch, -1, -1)
        for idx in range(base_seq):
            out = self._apply_forward(fn_input[:, idx], out, output_list, output, top, positional_encoding[:, idx])
            top = False
        for idx in range(base_seq, seq):
            out = self._apply_forward(torch.zeros(batch, input_features, device=fn_input.device),
                                      out, output_list, output, top, positional_encoding[:, idx])
            top = False
        if self.return_sequences:
            out = torch.stack(output[self.delay:], 1)
        return out


class Transpose(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input.transpose(1, 2)
