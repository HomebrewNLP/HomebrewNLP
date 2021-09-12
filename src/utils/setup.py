import math
import random
import typing

import numpy as np
import torch
from deepspeed.runtime import lr_schedules

from src.dataclass import Context
from src.model import LinearAttention, Trainer
from src.optimizers.build import build_optimizer
from src.utils.formatting import pretty_print


def setup_torch(seed: int):
    torch._C._debug_set_autodiff_subgraph_inlining(False)  # skipcq: PYL-W0212
    torch._C._set_graph_executor_optimize(True)  # skipcq: PYL-W0212
    torch._C._set_backcompat_broadcast_warn(False)  # skipcq: PYL-W0212
    torch._C._set_backcompat_keepdim_warn(False)  # skipcq: PYL-W0212
    torch._C._set_cudnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_mkldnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_mkldnn_enabled(True)  # skipcq: PYL-W0212
    torch._C._set_cudnn_benchmark(True)  # skipcq: PYL-W0212
    torch._C._set_cudnn_deterministic(False)  # skipcq: PYL-W0212
    torch._C._set_cudnn_allow_tf32(True)  # skipcq: PYL-W0212
    torch._C._set_cublas_allow_tf32(True)  # skipcq: PYL-W0212
    torch._C._jit_set_inline_everything_mode(True)  # skipcq: PYL-W0212

    torch._C._jit_set_profiling_executor(True)  # skipcq: PYL-W0212
    torch._C._jit_set_profiling_mode(True)  # skipcq: PYL-W0212
    torch._C._jit_override_can_fuse_on_cpu(False)  # skipcq: PYL-W0212
    torch._C._jit_override_can_fuse_on_gpu(True)  # skipcq: PYL-W0212
    torch._C._jit_set_texpr_fuser_enabled(True)  # skipcq: PYL-W0212
    torch._C._jit_set_nvfuser_enabled(False)  # skipcq: PYL-W0212

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model(ctx: Context, load_model: bool) -> typing.Tuple[Trainer, torch.optim.Optimizer, lr_schedules.OneCycle]:
    mod = LinearAttention(ctx).to(dtype=torch.float16 if ctx.model.float16 else torch.float)
    optimizer = build_optimizer(ctx, mod.parameters())
    scheduler = lr_schedules.OneCycle(optimizer,
                                      ctx.optimizer.one_cycle.cycle_min_lr,
                                      ctx.optimizer.one_cycle.cycle_max_lr,
                                      ctx.optimizer.one_cycle.decay_lr_rate,
                                      ctx.optimizer.one_cycle.cycle_first_step_size,
                                      ctx.optimizer.one_cycle.cycle_second_step_size,
                                      ctx.optimizer.one_cycle.cycle_first_stair_count,
                                      ctx.optimizer.one_cycle.cycle_second_stair_count,
                                      ctx.optimizer.one_cycle.decay_step_size,
                                      ctx.optimizer.one_cycle.cycle_momentum,
                                      ctx.optimizer.one_cycle.cycle_min_mom,
                                      ctx.optimizer.one_cycle.cycle_max_mom,
                                      ctx.optimizer.one_cycle.decay_mom_rate,
                                      ctx.optimizer.one_cycle.last_batch_iteration)
    mod = Trainer(ctx, mod, optimizer, scheduler)

    if ctx.model.print_on_init:
        pretty_print(str(mod))

    parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, mod.parameters()))
    base = int(math.log10(parameters) / 3)
    pretty_print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')
    if load_model:
        mod.load()
    if not ctx.model.offloading:
        mod = mod.to(ctx.model.device)
    return mod, optimizer, scheduler


def encode(prompt: str) -> torch.Tensor:
    return torch.as_tensor(np.frombuffer(prompt.encode('UTF-8'), np.uint8))


def decode(output: torch.LongTensor) -> str:
    return ''.join(chr(c) for c in output.view(-1).unbind(0))
