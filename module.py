import torch


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    return fn_input * torch.tanh(torch.nn.functional.softplus(fn_input))


class ReversibleRNNFunction(torch.autograd.Function):
    @staticmethod
    def _calc(fn_input, sequence_input, bn_weight, bn_bias, linear_param):
        std, mean = torch.std_mean(fn_input)
        out = torch.nn.functional.batch_norm(fn_input, mean, std, bn_weight, bn_bias)
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
        out = torch.cat(out, -1)
        return out

    @staticmethod
    def forward(ctx, fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1,
                output_list):
        ctx.save_for_backward(sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1)
        ctx.output_list = output_list
        if output_list:
            output_list.clear()
        with torch.no_grad():
            out = ReversibleRNNFunction._forward_pass(fn_input, sequence_input, linear_param0, linear_param1,
                                                      bn_weight0, bn_bias0, bn_weight1, bn_bias1)

        out.requires_grad_(sequence_input.requires_grad)
        output_list.append(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1 = ctx.saved_tensors
        if not sequence_input.requires_grad:
            return (None,) * 9
        out0, out1 = ctx.output_list[0].chunk(2, -1)
        with torch.no_grad():
            inp0 = out0 - ReversibleRNNFunction._calc(out1, sequence_input, bn_weight1, bn_bias1, linear_param1)
            inp1 = out1 - ReversibleRNNFunction._calc(inp0, sequence_input, bn_weight0, bn_bias0, linear_param0)
        fn_input = torch.cat([inp0, inp1], -1)
        fn_input.detach_()
        args = (fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1)
        grad_out = ReversibleRNNFunction._forward_pass(*args)
        return torch.autograd.grad(grad_out, args, grad_output) + (None,)


class RevRNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, sequence_dim=1, return_sequences=False):
        super(RevRNN, self).__init__()
        self.linear_param0 = torch.nn.Parameter(torch.zeros((input_features + hidden_features, hidden_features)))
        self.linear_param1 = torch.nn.Parameter(torch.zeros((input_features + hidden_features, hidden_features)))
        self.bn_weight0 = torch.nn.Parameter(torch.ones(hidden_features))
        self.bn_weight1 = torch.nn.Parameter(torch.ones(hidden_features))
        self.bn_bias0 = torch.nn.Parameter(torch.zeros(hidden_features))
        self.bn_bias1 = torch.nn.Parameter(torch.zeros(hidden_features))
        self.hidden_state = torch.nn.Parameter(torch.zeros(hidden_features))
        self.sequence_dim = sequence_dim
        self.return_sequences = return_sequences

    def forward(self, fn_input: torch.Tensor):
        output_list = []
        out = self.hidden_state
        if self.return_sequences:
            output = []
        for itm in fn_input.unbind(self.sequence_dim):
            out = ReversibleRNNFunction.apply(out, itm, self.linear_param0, self.linear_param1, self.bn_weight0,
                                              self.bn_bias0, self.bn_weight1, self.bn_weight1, output_list)
            if self.return_sequences:
                output.append(out)
        if self.return_sequences:
            out = torch.cat(output, -1)
        return out
