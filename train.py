import math
import time

import deepspeed
import numpy as np
import torch

import module

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

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_nvfuser_enabled(False)

HIDDEN = 16  # hidden units are squared
DELAY = 64
BATCH_SIZE = 128
CHECKPOINT = 1
SEQUENCE_LENGTH = 2 ** 8
PRINTERVALL = 32
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float  # torch.double

BATCH_SIZE *= CHECKPOINT
CONFIG = {"train_batch_size": BATCH_SIZE,
          "gradient_accumulation_steps": CHECKPOINT,
          "optimizer": {"type": "Adam", "params": {"lr": 3e-4}},
          "fp16": {"enabled": False},
          "zero_optimization": {
              "stage": 3,
              "cpu_offload": True,
              "contiguous_gradients": False,
              "overlap_comm": True,
              "offload_param": {"device": "cpu",
                                "pin_memory": True},
              "offload_optimizer": {"device": "cpu",
                                    "pin_memory": True},
              "stage3_max_live_parameter s": 1e8,
              "stage3_max_reuse_distance": 1e8,
              "stage3_prefetch_bucket_size": 5e7,
              "stage3_param_persistence_threshold": 1e6,
              "elastic_checkpoint": True},
          "activation_checkpointing": {"cpu_checkpointing": True, "contiguous_memory_optimization": True}
          }


# 3.66 @ 256

def parameter_count(net):
    return sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters()))


def init(module: torch.nn.Module):
    if hasattr(module, "weight") and hasattr(module.weight, "data"):
        if "norm" in module.__class__.__name__.lower() or (
                hasattr(module, "__str__") and "norm" in str(module).lower()):
            torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
        else:
            # torch.nn.init.constant_(module.weight.data, 0)
            torch.nn.init.orthogonal_(module.weight.data)
    if hasattr(module, "bias") and hasattr(module.bias, "data"):
        torch.nn.init.constant_(module.bias.data, 0)


mod = module.LinearAttention(256,
                             HIDDEN,
                             256,
                             delay=DELAY,
                             input_count=SEQUENCE_LENGTH).to(device=DEVICE, dtype=DTYPE)
mod.apply(init)
print(mod)
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

mean_loss = 0
curr_loss = 0
mean_acc = 0
# opt = shampoo.Shampoo(mod.parameters(), 0.1)
mod, opt, _, _ = deepspeed.initialize(model=mod, config=CONFIG, model_parameters=mod.parameters())

# torch.autograd.set_detect_anomaly(True)
while True:
    start_time = time.time()
    for i in range(1, 1 + length):
        src = tensor[batch_index].to(DEVICE)
        tgt = tensor[batch_index + DELAY + 1].to(DEVICE)
        lss = mod(src.to(DEVICE), tgt.to(DEVICE))
        mod.backward(lss)
        # lss.backward()
        with torch.no_grad():
            mod.step()
            # opt.step()
            # opt.zero_grad()
            curr_loss += lss.detach()
            batch_index += SEQUENCE_LENGTH
            if i % PRINTERVALL == 0:
                mean_loss += curr_loss
                print(
                    f"[{i:{len_len}d}/{length}] Loss: {curr_loss.item() / PRINTERVALL:7.4f} - "
                    f"Mean: {mean_loss.item() / i:7.4f} | "
                    f"LR: {opt.param_groups[0]['lr']:.6f}"
                    f" | Batch/s: {i / (time.time() - start_time):.3f}")
                curr_loss = 0

# seq 1024
# no lr scheduler: [   8256/1262160] Loss:  5.5985 - Mean:  8.6705 | Acc:   2.93% - Mean:   2.00% | Batch/s: 1.863s
# lr sched [  9728/631079] Loss:  5.5599 - Mean:  5.6732 | Acc:   2.64% - Mean:   3.19% - LR: 4.096000000000002e-05 | Batch/s: 1
# seq 64
# [   10496/10097290] Loss:  5.6123 - Mean:  6.0518 | Acc:   3.12% - Mean:   4.25% - LR: 0.000041 | Batch/s: 24.568s
# hidden=64
# [   11008/10097290] Loss:  6.3630 - Mean: 10.2268 | Acc:   0.00% - Mean:   4.60% - LR: 0.000016 | Batch/s: 15.515s
# relu instead of orthogonal activation
# [   10880/10097290] Loss:  6.2875 - Mean: 10.2672 | Acc:   6.25% - Mean:   4.80% - LR: 0.000016 | Batch/s: 15.859s


baseline = """
[     32/2178794] Loss:  5.6058 - Mean:  5.6058 | LR: 0.001000 | Batch/s: 0.931
[     64/2178794] Loss:  4.7250 - Mean:  5.1654 | LR: 0.001000 | Batch/s: 0.991
[     96/2178794] Loss:  4.2278 - Mean:  4.8529 | LR: 0.001000 | Batch/s: 1.002
[    128/2178794] Loss:  3.9259 - Mean:  4.6211 | LR: 0.001000 | Batch/s: 0.994
[    160/2178794] Loss:  3.7271 - Mean:  4.4423 | LR: 0.001000 | Batch/s: 0.991
[    192/2178794] Loss:  3.5590 - Mean:  4.2951 | LR: 0.001000 | Batch/s: 0.983
[    224/2178794] Loss:  3.4259 - Mean:  4.1709 | LR: 0.001000 | Batch/s: 0.980
[    256/2178794] Loss:  3.5348 - Mean:  4.0914 | LR: 0.001000 | Batch/s: 0.982
[    288/2178794] Loss:  3.6209 - Mean:  4.0391 | LR: 0.001000 | Batch/s: 0.980
[    320/2178794] Loss:  3.2985 - Mean:  3.9651 | LR: 0.001000 | Batch/s: 0.983
[    352/2178794] Loss:  3.2080 - Mean:  3.8962 | LR: 0.001000 | Batch/s: 0.986
[    384/2178794] Loss:  3.2278 - Mean:  3.8405 | LR: 0.001000 | Batch/s: 0.993
[    416/2178794] Loss:  3.2473 - Mean:  3.7949 | LR: 0.001000 | Batch/s: 0.999
[    448/2178794] Loss:  3.1820 - Mean:  3.7511 | LR: 0.001000 | Batch/s: 1.004
[    480/2178794] Loss:  3.3277 - Mean:  3.7229 | LR: 0.001000 | Batch/s: 1.009
[    512/2178794] Loss:  3.3316 - Mean:  3.6984 | LR: 0.001000 | Batch/s: 1.013
[    544/2178794] Loss:  3.1846 - Mean:  3.6682 | LR: 0.001000 | Batch/s: 1.016
[    576/2178794] Loss:  3.3547 - Mean:  3.6508 | LR: 0.001000 | Batch/s: 1.019
[    608/2178794] Loss:  3.5244 - Mean:  3.6441 | LR: 0.001000 | Batch/s: 1.022
[    640/2178794] Loss:  3.3821 - Mean:  3.6310 | LR: 0.001000 | Batch/s: 1.025
[    672/2178794] Loss:  3.2560 - Mean:  3.6132 | LR: 0.001000 | Batch/s: 1.027
[    704/2178794] Loss:  3.2294 - Mean:  3.5957 | LR: 0.001000 | Batch/s: 1.029
[    736/2178794] Loss:  3.3797 - Mean:  3.5863 | LR: 0.001000 | Batch/s: 1.031
[    768/2178794] Loss:  3.6027 - Mean:  3.5870 | LR: 0.001000 | Batch/s: 1.033
[    800/2178794] Loss:  3.4204 - Mean:  3.5804 | LR: 0.001000 | Batch/s: 1.030
[    832/2178794] Loss:  3.2638 - Mean:  3.5682 | LR: 0.001000 | Batch/s: 1.027
[    864/2178794] Loss:  3.2533 - Mean:  3.5565 | LR: 0.001000 | Batch/s: 1.028
[    896/2178794] Loss:  3.4802 - Mean:  3.5538 | LR: 0.001000 | Batch/s: 1.028
[    928/2178794] Loss:  3.6548 - Mean:  3.5573 | LR: 0.001000 | Batch/s: 1.028
[    960/2178794] Loss:  3.5224 - Mean:  3.5561 | LR: 0.001000 | Batch/s: 1.028
[    992/2178794] Loss:  3.2971 - Mean:  3.5478 | LR: 0.001000 | Batch/s: 1.028
[   1024/2178794] Loss:  3.1801 - Mean:  3.5363 | LR: 0.001000 | Batch/s: 1.028
[   1056/2178794] Loss:  3.3553 - Mean:  3.5308 | LR: 0.001000 | Batch/s: 1.024
[   1088/2178794] Loss:  3.3954 - Mean:  3.5268 | LR: 0.001000 | Batch/s: 1.024
[   1120/2178794] Loss:  3.5502 - Mean:  3.5275 | LR: 0.001000 | Batch/s: 1.026
[   1152/2178794] Loss:  3.2749 - Mean:  3.5205 | LR: 0.001000 | Batch/s: 1.026
[   1184/2178794] Loss:  3.2293 - Mean:  3.5126 | LR: 0.001000 | Batch/s: 1.026
[   1216/2178794] Loss:  3.1798 - Mean:  3.5038 | LR: 0.001000 | Batch/s: 1.028
[   1248/2178794] Loss:  3.1897 - Mean:  3.4958 | LR: 0.001000 | Batch/s: 1.028
[   1280/2178794] Loss:  3.2054 - Mean:  3.4885 | LR: 0.001000 | Batch/s: 1.027
"""
