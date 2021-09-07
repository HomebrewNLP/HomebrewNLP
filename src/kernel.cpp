#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> norm(std::vector<torch::Tensor> chunks) {
    torch::Tensor inp = chunks[0] * chunks[1] + chunks[2];
    inp -= inp.mean(1, true);
    torch::Tensor std = 1 / (torch::norm(inp, 2, 1, true) * sqrt(1 / inp.size(1)) + 1e-5);
    return {torch::leaky_relu(inp * std, 0.02), chunks[0], chunks[1], std};
}

std::vector<torch::Tensor> norm_backward(torch::Tensor out,
                                         torch::Tensor chunk0,
                                         torch::Tensor chunk1,
                                         torch::Tensor std,
                                         torch::Tensor d_out) {
    d_out = torch::leaky_relu(d_out, 0.02);
    out = torch::leaky_relu(out, 1 / 0.02);
    d_out = (d_out - out * (d_out * out).mean(1, true)) * std;
    d_out -= d_out.mean(1, true);
    return {d_out * chunk1, d_out * chunk0, d_out};
}


std::vector<torch::Tensor> _forward(torch::Tensor x1,
                                   torch::Tensor w0,
                                   torch::Tensor w1,
                                   torch::Tensor w2) {
    x1 = torch::conv1d(x1, w0);
    std::vector<torch::Tensor> chunks = x1.chunk(3, 1);
    at::TensorOptions opt = at::TensorOptions(x1.dtype()).device(x1.device());
    torch::Tensor divisor = torch::arange(x1.size(2), opt).unsqueeze(0).unsqueeze(0);
    chunks[0] = torch::cumsum(chunks[0], 2) / divisor;
    std::vector<torch::Tensor> intermediate0 = norm(chunks);
    at::IntArrayRef pad = {w1.size(2) - 1, 0};
    x1 = torch::conv1d(torch::constant_pad_nd(intermediate0[0], pad), w1);
    std::vector<torch::Tensor> intermediate1 = norm(x1.chunk(3, 1));
    x1 = torch::conv1d(intermediate1[0], w2);
    intermediate0.insert(intermediate0.end(), intermediate1.begin(), intermediate1.end());
    intermediate0.push_back(x1);
    return intermediate0;
}


torch::Tensor forward(torch::Tensor x0,
                      torch::Tensor x1,
                      torch::Tensor w0,
                      torch::Tensor w1,
                      torch::Tensor w2) {
    std::vector<torch::Tensor> out = _forward(x1, w0, w1, w2);
    return out[8] + x0;
}

std::vector<torch::Tensor> backward(torch::Tensor y1,
                                    torch::Tensor x1,
                                    torch::Tensor dy,
                                    torch::Tensor w0,
                                    torch::Tensor w1,
                                    torch::Tensor w2) {
    std::vector<torch::Tensor> out = _forward(x1, w0, w1, w2);
    torch::Tensor intermediate0 = out[0];
    torch::Tensor chunk00 = out[1];
    torch::Tensor chunk01 = out[2];
    torch::Tensor std0 = out[3];
    torch::Tensor intermediate1 = out[4];
    torch::Tensor chunk10 = out[5];
    torch::Tensor chunk11 = out[6];
    torch::Tensor std1 = out[7];
    int batch = x1.size(0);
    int sequence = x1.size(2);
    int features = w1.size(1);
    int kernel = w1.size(2);
    torch::Tensor d_tmp = torch::conv1d(dy, w2.transpose(0, 1));
    w2 = torch::einsum("boh,bih->oi", {dy, intermediate1}).unsqueeze_(2);
    d_tmp = torch::cat(norm_backward(intermediate1, chunk10, chunk11, std1, d_tmp), 1);
    at::IntArrayRef pad = {kernel - 1, 0};
    d_tmp = torch::constant_pad_nd(d_tmp, pad);
    torch::Tensor tmp = intermediate0.view({batch, features, 1, sequence}).transpose_(0, 1);
    torch::Tensor d_w1 = torch::conv2d(d_tmp.view({1, batch, features * 3, sequence + kernel - 1}), tmp);
    d_tmp = torch::conv1d(d_tmp, w1.transpose(0, 1));
    std::vector<torch::Tensor> d_norm = norm_backward(intermediate0, chunk00, chunk01, std0, d_tmp);
    at::TensorOptions opt = at::TensorOptions(x1.dtype()).device(x1.device());
    d_norm[0] = d_norm[0].cumsum(2) / torch::arange(sequence, opt).view({1, 1, sequence});
    d_tmp = torch::cat(d_norm, 1);
    torch::Tensor d_w0 = torch::einsum("boh,bih->oi", {d_tmp, x1}).unsqueeze_(2);
    d_tmp = torch::conv1d(d_tmp, w0.transpose(0, 1));
    return {y1 - out[8], d_w0, d_w1.squeeze(0).transpose_(0, 1), w2, d_tmp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "forward");
    m.def("backward", &backward, "backward");
}
