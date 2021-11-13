import torch
import wandb

from src.dataclass import Context
from src.dataset import get_dataset
from src.utils.formatting import WandbLog
from src.utils.setup import get_model


def train_model(ctx: Context, steps=None, load_model: bool = False):
    wandb.init(project=ctx.log.wandb.project, entity=ctx.log.wandb.entity, config=ctx.serialize())
    ctx = Context(wandb.config)

    data = get_dataset(ctx)
    data_len = len(data)
    mod = get_model(ctx, load_model, next(data)[0])
    wandb.watch(mod, log=ctx.log.wandb.model_log_type, log_freq=ctx.log.wandb.log_frequency)

    log = WandbLog(ctx, data_len)
    mean_loss = torch.zeros([], device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)
    mean_max_loss = mean_loss.clone()

    i = 0
    while True:
        i += 1

        mean_loss += mod.accumulated_step(next(data))
        if ctx.optimizer.sharpness_aware_minimization.enabled:
            with torch.no_grad():
                for p in mod.gradients():
                    if ctx.optimizer.sharpness_aware_minimization.adaptive:
                        p.grad *= p.square()
                    p.grad *= ctx.optimizer.sharpness_aware_minimization.step_size
                    p.add_(p.grad)
                    p.prev_step = p.grad
                    p.grad = None
            mean_max_loss += mod.accumulated_step(next(data))
        mod.optimizer.step()
        if ctx.optimizer.sharpness_aware_minimization.enabled:
            with torch.no_grad():
                for p in mod.gradients():
                    p.sub_(p.prev_step)
                    p.prev_step = None
                    p.grad = None
        else:
            mod.zero_grad()
        mod.scheduler.step()
        for p in mod.optimizer.param_groups:  # OneCycle resets beta2 to 0.990
            p['betas'] = p['betas'][0], mod.ctx.optimizer.beta2
        with torch.no_grad():
            if mod.ctx.log.loss_steps_per_print and i % mod.ctx.log.loss_steps_per_print == 0:
                log(mean_loss, mean_max_loss,
                    mod.optimizer.param_groups[0]['lr'], mod.optimizer.param_groups[0]['betas'])
                mean_loss.zero_()
                mean_max_loss.zero_()
            if mod.ctx.model.steps_per_checkpoint and i % mod.ctx.model.steps_per_checkpoint == 0:
                mod.save()
        if steps and i > steps:
            return
