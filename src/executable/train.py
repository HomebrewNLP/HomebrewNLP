import torch
import wandb

from src.dataclass import Context
from src.dataset import get_dataset
from src.model import sorted_weight_values, sorted_weights
from src.utils.formatting import WandbLog
from src.utils.setup import get_model


def train_model(ctx: Context, steps=None, load_model: bool = False):
    wandb.init(project=ctx.log.wandb.project, entity=ctx.log.wandb.entity, config=ctx.serialize())
    ctx = Context(wandb.config)

    data = get_dataset(ctx)
    log = WandbLog(ctx, len(data))

    mod, opt, sched = get_model(ctx, load_model)
    itr = iter(data)
    mod = torch.jit.trace(mod, next(itr), check_trace=False)
    wandb.watch(mod, log=ctx.log.wandb.model_log_type, log_freq=ctx.log.wandb.log_frequency)

    mean_loss0 = torch.zeros([], device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)
    mean_loss1 = torch.zeros([], device=ctx.model.device, dtype=torch.float16 if ctx.model.float16 else torch.float)

    i = 0
    while True:
        i += 1
        loss0, loss1, weights = mod(*next(itr))
        mean_loss0 += loss0
        mean_loss1 += loss1
        mod.load_state_dict({k: s for s, (k, v) in zip(weights, sorted_weights(mod))})
        with torch.no_grad():
            if ctx.log.loss_steps_per_print and i % ctx.log.loss_steps_per_print == 0:
                log(mean_loss0, mean_loss1, opt.param_groups[0]['lr'], opt.param_groups[0]['betas'])
                mean_loss0.zero_()
                mean_loss1.zero_()
            if ctx.model.steps_per_checkpoint and i % ctx.model.steps_per_checkpoint == 0:
                mod.save()
        if steps and i > steps:
            return
