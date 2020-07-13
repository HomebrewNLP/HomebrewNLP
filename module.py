import torch


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    return fn_input * torch.tanh(torch.nn.functional.softplus(fn_input))


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def _calc(fn_input, sequence_input, bn_weight, bn_bias, linear_param):
        std, mean = torch.std_mean(fn_input, 0, keepdim=True)
        out = (fn_input - mean) / std * bn_weight + bn_bias
        out = mish(out)
        out = torch.nn.functional.linear(torch.cat([out, sequence_input], -1), linear_param, None)
        return out

    @staticmethod
    def _forward_pass(fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1,
                      bn_bias1):
        inp = fn_input.chunk(2, -1)
        out = inp[0]
        params = [[bn_weight0, bn_bias0, linear_param0], [bn_weight1, bn_bias1, linear_param1]]
        outputs = [None, None]
        for i in range(2):
            outputs[1 - i] = out = ReversibleRNNFunction._calc(out, sequence_input, *params[i]) + inp[1 - i]
        out = torch.cat(outputs, -1)
        return out

    @staticmethod
    def forward(ctx, fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1,
                output_list, top):
        ctx.save_for_backward(sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1)
        ctx.output_list = output_list
        ctx.top = top
        if output_list:
            output_list.clear()
        with torch.no_grad():
            out = ReversibleRNNFunction._forward_pass(fn_input, sequence_input, linear_param0, linear_param1,
                                                      bn_weight0, bn_bias0, bn_weight1, bn_bias1)
        with torch.enable_grad():
            out.requires_grad_(True)
            output_list.append(out)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1 = ctx.saved_tensors
        if not sequence_input.requires_grad:
            return (None,) * 10
        out0, out1 = ctx.output_list.pop(0).chunk(2, -1)
        with torch.no_grad():
            inp0 = out0 - ReversibleRNNFunction._calc(out1, sequence_input, bn_weight1, bn_bias1, linear_param1)
            inp1 = out1 - ReversibleRNNFunction._calc(inp0, sequence_input, bn_weight0, bn_bias0, linear_param0)
        with torch.enable_grad():
            fn_input = torch.cat([inp0, inp1], -1)
            fn_input.detach_()
            fn_input.requires_grad_(True)
            args = (fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1)
            grad_out = ReversibleRNNFunction._forward_pass(*args)
        grad_out.requires_grad_(True)
        grad_out = torch.autograd.grad(grad_out, args, grad_output)
        fn_input.detach_()
        fn_input.requires_grad_(True)
        if not ctx.top:
            ctx.output_list.append(fn_input)
        return grad_out + (None, None)


class RevRNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, sequence_dim=1, return_sequences=False, delay=8):
        super(RevRNN, self).__init__()
        if hidden_features % 2:
            raise UserWarning(f"Ignoring uneven hidden feature and proceeding as if equal {hidden_features // 2 * 2}")
        hidden_features = hidden_features // 2
        self.linear_param0 = torch.nn.Parameter(torch.zeros((hidden_features, input_features + hidden_features)))
        self.linear_param1 = torch.nn.Parameter(torch.zeros((hidden_features, input_features + hidden_features)))
        self.bn_weight0 = torch.nn.Parameter(torch.ones(hidden_features))
        self.bn_weight1 = torch.nn.Parameter(torch.ones(hidden_features))
        self.bn_bias0 = torch.nn.Parameter(torch.zeros(hidden_features))
        self.bn_bias1 = torch.nn.Parameter(torch.zeros(hidden_features))
        self.hidden_state = torch.nn.Parameter(torch.zeros(hidden_features * 2))

        self.sequence_dim = sequence_dim
        self.return_sequences = return_sequences
        self.delay = delay
        self.input_features = input_features

    def _apply_forward(self, itm, out, output_list, function_output, top):
        out = ReversibleRNNFunction.apply(out, itm, self.linear_param0, self.linear_param1, self.bn_weight0,
                                          self.bn_bias0, self.bn_weight1, self.bn_weight1, output_list, top)
        if self.return_sequences:
            function_output.append(out)
        return out

    def forward(self, fn_input: torch.Tensor):
        output_list = []
        batch = fn_input.size(0)
        out = self.hidden_state.view(1, -1).expand(batch, -1)
        out.requires_grad_(fn_input.requires_grad)
        input_features = self.input_features
        output = []
        top = True

        for idx in range(fn_input.size(self.sequence_dim)):
            out = self._apply_forward(fn_input[:, idx], out, output_list, output, top)
            top = False
        for _ in range(self.delay):
            out = self._apply_forward(torch.zeros(batch, input_features), out, output_list, output, top)
            top = False
        if self.return_sequences:
            out = torch.cat(output[self.delay:], -1)
        return out
