import time

import deepspeed
import torch

from src import model
from src.dataclass import Context
from src.utils import get_deepspeed_config


def main(ctx: Context):
    dtype = torch.float16 if ctx.model.float16 else torch.float
    config = get_deepspeed_config(ctx)

    mod = model.LinearAttention(ctx)
    mod = mod.to(dtype=dtype)

    tensor = torch.load(ctx.dataset.file_name)
    tensor = tensor.long()

    batch_index = torch.arange(0, ctx.model.batch_size * ctx.optimizer.gradient_accumulation_steps).view(-1, 1)
    item_index = torch.arange(0, ctx.model.sequence_length).view(1, -1)
    batch_index = batch_index + item_index

    length = tensor.size(0) // ctx.model.sequence_length - 1
    len_len = len(str(length))

    mean_loss = 0
    curr_loss = 0
    mod, opt, _, lr_scheduler = deepspeed.initialize(model=mod, config=config, model_parameters=mod.parameters())

    while True:
        start_time = time.time()
        for i in range(1, 1 + length):
            src = tensor[batch_index].to(ctx.model.device)
            tgt = tensor[batch_index + 1].to(ctx.model.device)
            lss = mod(src.to(ctx.model.device), tgt.to(ctx.model.device))
            mod.backward(lss)
            with torch.no_grad():
                mod.step()
                lr_scheduler.step()
                curr_loss += lss.detach()
                batch_index += ctx.model.sequence_length
                if i % ctx.log.loss_steps_per_print == 0:
                    mean_loss += curr_loss
                    print(f"[{i:{len_len}d}/{length}] Loss: {curr_loss.item() / ctx.log.loss_steps_per_print:7.4f} -",
                          f"Mean: {mean_loss.item() / i:7.4f} |",
                          f"LR: {opt.param_groups[0]['lr']:.6f}",
                          f"| Batch/s: {i / (time.time() - start_time):.3f}")
                    curr_loss = 0
