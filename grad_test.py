import torch

import module

BATCH = 2
IN_FEATURES = 2
OUT_FEATURES = 2

mod = module.RevRNN(IN_FEATURES, OUT_FEATURES)
# fn_input, sequence_input, linear_param0, linear_param1, bn_weight0, bn_bias0, bn_weight1, bn_bias1,
#                output_list, top, pos_enc
mod = mod.double()
pos = torch.randn(BATCH, 2, requires_grad=False, dtype=torch.double)
inp = torch.randn(BATCH, OUT_FEATURES, requires_grad=True, dtype=torch.double)
out = torch.randn(BATCH, IN_FEATURES, requires_grad=True, dtype=torch.double)
inp.requires_grad_(True)
out.requires_grad_(True)
torch.autograd.gradcheck(module.ReversibleRNNFunction.apply, (inp,
                                                              out,
                                                              mod.linear_param00,
                                                              mod.linear_param10,
                                                              [],
                                                              True,
                                                              pos),
                         atol=1e-4)
