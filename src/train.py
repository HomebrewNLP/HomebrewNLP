import math
import time

import deepspeed
import numpy as np
import torch

from src import module
from src.dataclass import Context

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


def main(ctx: Context):
    dtype = torch.float16 if ctx.model.float16 else torch.float

    config = {"train_batch_size": ctx.model.batch_size * ctx.train.gradient_accumulation_steps,
              "gradient_accumulation_steps": ctx.train.gradient_accumulation_steps,
              "optimizer": {"type": "Adam", "params": {"lr": ctx.optimizer.learning_rate}},
              "fp16": {"enabled": ctx.model.float16},
              "zero_optimization": {"stage": 3,
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

    mod = module.LinearAttention(ctx.dataset.classes,
                                 ctx.model.features,
                                 ctx.dataset.classes,
                                 depth=ctx.model.depth,
                                 input_count=ctx.model.sequence_length,
                                 weight_shared_blocks=ctx.model.weight_shared_blocks)
    mod = mod.to(device=ctx.model.device, dtype=dtype)
    mod.apply(init)
    print(mod)
    parameters = parameter_count(mod)
    base = int(math.log10(parameters) / 3)
    print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')

    tensor = torch.load('out.tensor')
    tensor = tensor.long()

    batch_index = torch.arange(0, ctx.model.batch_size * ctx.train.gradient_accumulation_steps).view(-1, 1)
    item_index = torch.arange(0, ctx.model.sequence_length).view(1, -1)
    batch_index = batch_index + item_index

    length = tensor.size(0) // ctx.model.sequence_length - 1
    len_len = len(str(length))

    mean_loss = 0
    curr_loss = 0
    mod, opt, _, _ = deepspeed.initialize(model=mod, config=config, model_parameters=mod.parameters())

    while True:
        start_time = time.time()
        for i in range(1, 1 + length):
            src = tensor[batch_index].to(ctx.model.device)
            tgt = tensor[batch_index + 1].to(ctx.model.device)
            lss = mod(src.to(ctx.model.device), tgt.to(ctx.model.device))
            mod.backward(lss)
            with torch.no_grad():
                mod.step()
                curr_loss += lss.detach()
                batch_index += ctx.model.sequence_length
                if i % ctx.train.print_interval == 0:
                    mean_loss += curr_loss
                    print(
                        f"[{i:{len_len}d}/{length}] Loss: {curr_loss.item() / ctx.train.print_interval:7.4f} - "
                        f"Mean: {mean_loss.item() / i:7.4f} | "
                        f"LR: {opt.param_groups[0]['lr']:.6f}"
                        f" | Batch/s: {i / (time.time() - start_time):.3f}")
                    curr_loss = 0
