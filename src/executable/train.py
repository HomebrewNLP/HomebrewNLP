import time

import deepspeed
import torch

from src.dataclass import Context
from src.dataset import get_dataset
from src.utils.formatting import pretty_print
from src.utils.setup import get_model, get_deepspeed_config


def train_model(ctx: Context, steps=None):
    mod = get_model(ctx)
    engine = deepspeed.DeepSpeedEngine(None,
                                       mod,
                                       model_parameters=mod.parameters(),
                                       config=get_deepspeed_config(ctx),
                                       dont_change_device=True)
    data = get_dataset(ctx)
    length = len(data)
    len_len = len(str(length))

    dtype = torch.float16 if ctx.model.float16 else torch.float
    mean_loss = torch.zeros([], device=ctx.model.device, dtype=dtype)
    curr_loss = torch.zeros([], device=ctx.model.device, dtype=dtype)
    start_time = time.time()
    for i, (src, tgt) in enumerate(data, 1):
        lss = engine(src.squeeze(0).to(device=ctx.model.device, non_blocking=True),
                     tgt.squeeze(0).to(device=ctx.model.device, non_blocking=True))
        engine.backward(lss)
        with torch.no_grad():
            engine.step()
            engine.lr_scheduler.step()
            curr_loss += lss.detach()
            if i % ctx.log.loss_steps_per_print == 0:
                mean_loss += curr_loss
                rate = i / (time.time() - start_time)
                pretty_print \
                    (f"[{i:{len_len}d}/{length}]",
                     f"Loss: {curr_loss.item() / ctx.log.loss_steps_per_print:7.4f} -",
                     f"Mean: {mean_loss.item() / i:7.4f} |",
                     f"LR: {engine.optimizer.param_groups[0]['lr']:.6f} |",
                     f"Batch/s: {rate:6.3f} -",
                     f"Tokens/day: {3600 * 24 * rate * ctx.model.batch_size * ctx.model.sequence_length:11,.0f}")
                curr_loss = 0
        if steps is not None and i > steps:
            return
