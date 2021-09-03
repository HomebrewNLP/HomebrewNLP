import torch
import wandb
from deepspeed.runtime import lr_schedules

from src.dataclass import Context
from src.dataset import get_dataset
from src.utils.formatting import WandbLog
from src.utils.setup import get_model


def clip_gradient(ctx: Context, mod: torch.nn.Module):
    for p in mod.parameters():
        if p.grad is None:
            continue

        g_norm = p.grad.norm(2, min(p.ndim - 1, 1), True).clamp(min=ctx.optimizer.agc.zero_division_eps)
        p_norm = p.norm(2, min(p.ndim - 1, 1), True).clamp(min=ctx.optimizer.agc.eps)
        grad_scale = (p_norm / g_norm * ctx.optimizer.agc.gradient_clipping).clamp(max=1)
        p.grad.data.copy_(p.grad * grad_scale)


def train_model(ctx: Context, steps=None, load_model: bool = False):
    wandb.init(project=ctx.log.wandb.project, entity=ctx.log.wandb.entity)
    wandb.config = ctx.serialize()
    ctx = Context(wandb.config)

    mod = get_model(ctx)
    wandb.watch(mod, log=ctx.log.wandb.model_log_type, log_freq=ctx.log.wandb.log_frequency)

    if load_model:
        mod.load()
    if not ctx.model.offloading:
        mod = mod.to(ctx.model.device)
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
    log = WandbLog(ctx, len(data))
    curr_loss = torch.zeros([], device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)

    for i, (src, tgt) in enumerate(data, 1):
        lss = mod(src.squeeze(0).to(device=ctx.model.device, non_blocking=True),
                  tgt.squeeze(0).to(device=ctx.model.device, non_blocking=True))
        lss.backward()
        with torch.no_grad():
            curr_loss += lss.detach()
            if i % ctx.optimizer.gradient_accumulation_steps:
                continue
            clip_gradient(ctx, mod)
            opt.step()
            opt.zero_grad()
            shed.step()
            if ctx.log.loss_steps_per_print and i % ctx.log.loss_steps_per_print == 0:
                log(curr_loss, opt.param_groups[0]['lr'])
                curr_loss = 0
            if ctx.model.steps_per_checkpoint and i % ctx.model.steps_per_checkpoint == 0:
                mod.save()
        if steps and i > steps:
            return
