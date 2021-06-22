import math
import time

import numpy as np
import torch

import module

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_nvfuser_enabled(True)
torch._C._debug_set_autodiff_subgraph_inlining(False)  # Not sure
torch._C._set_graph_executor_optimize(True)
torch._C._set_backcompat_broadcast_warn(False)
torch._C._set_backcompat_keepdim_warn(False)
torch._C._set_cudnn_enabled(True)
torch._C._set_mkldnn_enabled(True)
torch._C._set_mkldnn_enabled(True)
torch._C._set_cudnn_benchmark(True)
torch._C._set_cudnn_deterministic(False)
torch._C._set_cudnn_allow_tf32(True)
torch._C._set_cublas_allow_tf32(True)
torch._C._jit_set_inline_everything_mode(True)
torch._C._jit_set_texpr_fuser_enabled(True)

HIDDEN = 64 # hidden units are squared
DELAY = 0
BATCH_SIZE = 2
SEQUENCE_LENGTH = 2 ** 14
PRINTERVALL = 1
DEPTH = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float  # torch.double


# 3.66 @ 256

def parameter_count(net):
    return sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters()))


def init(module: torch.nn.Module):
    if hasattr(module, "weight") and hasattr(module.weight, "data"):
        if "norm" in module.__class__.__name__.lower() or (
                hasattr(module, "__str__") and "norm" in str(module).lower()):
            torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
        else:
            torch.nn.init.orthogonal_(module.weight.data)
    if hasattr(module, "bias") and hasattr(module.bias, "data"):
        torch.nn.init.constant_(module.bias.data, 0)


mod = module.FixedRevRNN(256,
                         HIDDEN,
                         256,
                         delay=DELAY,
                         return_sequences=True,
                         depth=DEPTH,
                         input_count=SEQUENCE_LENGTH).to(DEVICE).to(DTYPE)
mod.apply(init)
parameters = parameter_count(mod)
base = int(math.log10(parameters) / 3)
print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')

tensor = torch.load('out.tensor')
tensor = tensor.long()

batch_index = torch.arange(0, BATCH_SIZE).view(-1, 1)
item_index = torch.arange(0, SEQUENCE_LENGTH).view(1, -1)
batch_index = batch_index + item_index
base_index = batch_index.clone()

length = tensor.size(0) // SEQUENCE_LENGTH - 1
len_len = len(str(length))

mod = torch.jit.trace(mod, tensor[batch_index].to(DEVICE))
opt = torch.optim.AdamW(mod.parameters(), lr=0.0625 * 0.5, weight_decay=2e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=256, factor=0.4)  # 1024

mean_loss = 0
curr_loss = 0
mean_acc = 0
torch.autograd.set_detect_anomaly(True)
while True:
    start_time = time.time()
    for i in range(1, 1 + length):
        src = tensor[batch_index].to(DEVICE)
        tgt = tensor[batch_index + DELAY + 1].to(DEVICE)
        out = mod(src.to(DEVICE))
        out.transpose_(1, 2)
        lss = torch.nn.functional.cross_entropy(out, tgt)
        lss.backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            curr_loss += lss.detach()
            batch_index += SEQUENCE_LENGTH
            if i % PRINTERVALL == 0:
                mean_loss += curr_loss
                acc = (tgt == out.argmax(1)).sum().detach() / tgt.numel() * 100
                mean_acc += acc
                print(f"[{i:{len_len}d}/{length}] Loss: {curr_loss.item() / PRINTERVALL:7.4f} - Mean: {mean_loss.item() / i:7.4f}"
                      f" | Acc: {acc.item():6.2f}% - Mean: {mean_acc.item() / (i / PRINTERVALL):6.2f}% - LR: {opt.param_groups[0]['lr']:.6f}"
                      f" | Batch/s: {i / (time.time() - start_time):.3f}s")
                curr_loss = 0
                sch.step(curr_loss)

# seq 1024
# no lr scheduler: [   8256/1262160] Loss:  5.5985 - Mean:  8.6705 | Acc:   2.93% - Mean:   2.00% | Batch/s: 1.863s
# lr sched [  9728/631079] Loss:  5.5599 - Mean:  5.6732 | Acc:   2.64% - Mean:   3.19% - LR: 4.096000000000002e-05 | Batch/s: 1
# seq 64
# [   10496/10097290] Loss:  5.6123 - Mean:  6.0518 | Acc:   3.12% - Mean:   4.25% - LR: 0.000041 | Batch/s: 24.568s
# hidden=64
# [   11008/10097290] Loss:  6.3630 - Mean: 10.2268 | Acc:   0.00% - Mean:   4.60% - LR: 0.000016 | Batch/s: 15.515s
# relu instead of orthogonal activation
# [   10880/10097290] Loss:  6.2875 - Mean: 10.2672 | Acc:   6.25% - Mean:   4.80% - LR: 0.000016 | Batch/s: 15.859s
