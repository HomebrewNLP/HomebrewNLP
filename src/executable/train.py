import time
import typing

import torch
from deepspeed.runtime import lr_schedules

from src.dataclass import Context
from src.dataset import get_dataset
from src.utils.formatting import pretty_print
from src.utils.setup import get_model
from src.model import LinearAttentionCell, ParameterStore



def clip_gradient(ctx: Context, mod: torch.nn.Module):
    for p in mod.parameters():
        if p.grad is None:
            continue

        g_norm = p.grad.norm(2, min(p.ndim - 1, 1), True).clamp(min=ctx.optimizer.agc.zero_division_eps)
        p_norm = p.norm(2, min(p.ndim - 1, 1), True).clamp(min=ctx.optimizer.agc.eps)
        grad_scale = (p_norm / g_norm * ctx.optimizer.agc.gradient_clipping).clamp(max=1)
        p.grad.data.copy_(p.grad * grad_scale)


def train_model(ctx: Context, steps=None):
    mod = get_model(ctx)
    if not ctx.model.offloading:
        mod = mod.apply(ctx.model.device)
    opt = torch.optim.AdamW(mod.parameters())
    shed = lr_schedules.OneCycle(opt,
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
    data = get_dataset(ctx)
    length = len(data)
    len_len = len(str(length))

    dtype = torch.float16 if ctx.model.float16 else torch.float
    mean_loss = torch.zeros([], device=ctx.model.device, dtype=dtype)
    curr_loss = torch.zeros([], device=ctx.model.device, dtype=dtype)

    start_time = time.time()
    for i, (src, tgt) in enumerate(data, 1):
        lss = mod(src.squeeze(0).to(device=ctx.model.device, non_blocking=True),
                  tgt.squeeze(0).to(device=ctx.model.device, non_blocking=True))
        lss.backward()
        with torch.no_grad():
            clip_gradient(ctx, mod)
            opt.step()
            opt.zero_grad()
            shed.step()
            curr_loss += lss.detach()
            if i % ctx.log.loss_steps_per_print == 0:
                mean_loss += curr_loss
                rate = i / (time.time() - start_time)
                pretty_print \
                    (f"[{i:{len_len}d}/{length}]",
                     f"Loss: {curr_loss.item() / ctx.log.loss_steps_per_print:7.4f} -",
                     f"Mean: {mean_loss.item() / i:7.4f} |",
                     f"LR: {opt.param_groups[0]['lr']:.6f} |",
                     f"Batch/s: {rate:6.3f} -",
                     f"Tokens/day: {3600 * 24 * rate * ctx.model.batch_size * ctx.model.sequence_length:11,.0f}")
                curr_loss = 0
        if steps is not None and i > steps:
            return
