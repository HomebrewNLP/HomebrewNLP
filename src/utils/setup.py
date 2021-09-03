import math
import random

import numpy as np

import torch
from src.dataclass import Context
from src.model import LinearAttention
from src.utils.formatting import print


def setup_torch(seed: int):
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_deepspeed_config(ctx: Context) -> dict:
    return {"train_batch_size": ctx.model.batch_size * ctx.optimizer.gradient_accumulation_steps,
            "gradient_accumulation_steps": ctx.optimizer.gradient_accumulation_steps,
            "optimizer": {"type": ctx.optimizer.type,
                          "params": {"betas": [0.9, ctx.optimizer.beta2],
                                     "eps": ctx.optimizer.epsilon,
                                     "weight_decay": ctx.optimizer.weight_decay
                                     }
                          },
            "fp16": {"enabled": ctx.model.float16},
            "zero_optimization": {"stage": 3,
                                  "cpu_offload": ctx.optimizer.zero.cpu_offload,
                                  "contiguous_gradients": ctx.optimizer.zero.contiguous_gradients,
                                  "overlap_comm": ctx.optimizer.zero.overlap_comm,
                                  "offload_param": {"device": ctx.optimizer.zero.offload_param.device,
                                                    "pin_memory": ctx.optimizer.zero.offload_param.pin_memory},
                                  "offload_optimizer": {"device": ctx.optimizer.zero.offload_optimizer.device,
                                                        "pin_memory": ctx.optimizer.zero.offload_optimizer.pin_memory},
                                  "stage3_max_live_parameters": ctx.optimizer.zero.stage3_max_live_parameters,
                                  "stage3_max_reuse_distance": ctx.optimizer.zero.stage3_max_reuse_distance,
                                  "stage3_prefetch_bucket_size": ctx.optimizer.zero.stage3_prefetch_bucket_size,
                                  "stage3_param_persistence_threshold": ctx.optimizer.zero.stage3_param_persistence_threshold,
                                  },
            "activation_checkpointing": {"cpu_checkpointing": True, "contiguous_memory_optimization": True},
            "steps_per_print": ctx.log.deepspeed_steps_per_print,
            "wall_clock_breakdown": ctx.log.wall_clock_breakdown,
            "dump_state": ctx.log.dump_state,
            "scheduler": {"type": "OneCycle",
                          "params": {"cycle_min_lr": ctx.optimizer.one_cycle.cycle_min_lr,
                                     "cycle_max_lr": ctx.optimizer.one_cycle.cycle_max_lr,
                                     "decay_lr_rate": ctx.optimizer.one_cycle.decay_lr_rate,
                                     "cycle_first_step_size": ctx.optimizer.one_cycle.cycle_first_step_size,
                                     "cycle_second_step_size": ctx.optimizer.one_cycle.cycle_second_step_size,
                                     "cycle_first_stair_count": ctx.optimizer.one_cycle.cycle_first_stair_count,
                                     "cycle_second_stair_count": ctx.optimizer.one_cycle.cycle_second_stair_count,
                                     "decay_step_size": ctx.optimizer.one_cycle.decay_step_size,
                                     "cycle_momentum": ctx.optimizer.one_cycle.cycle_momentum,
                                     "cycle_min_mom": ctx.optimizer.one_cycle.cycle_min_mom,
                                     "cycle_max_mom": ctx.optimizer.one_cycle.cycle_max_mom,
                                     "decay_mom_rate": ctx.optimizer.one_cycle.decay_mom_rate,
                                     "last_batch_iteration": ctx.optimizer.one_cycle.last_batch_iteration
                                     }
                          }
            }


def get_model(ctx: Context) -> LinearAttention:
    mod = LinearAttention(ctx).to(dtype=torch.float16 if ctx.model.float16 else torch.float)

    if ctx.model.print_on_init:
        print(str(mod))

    parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, mod.parameters()))
    base = int(math.log10(parameters) / 3)
    print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')
    return mod


def encode(prompt: str) -> torch.LongTensor:
    return torch.as_tensor(np.frombuffer(prompt.encode('UTF-8'), np.uint8))


def decode(output: torch.LongTensor) -> str:
    return ''.join(chr(c) for c in output)
