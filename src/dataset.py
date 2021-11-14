import multiprocessing
import random
import typing

import torch
import torch.utils.data

from src.dataclass import Context


@torch.jit.script
def get_sample(data: torch.Tensor, batch_index: torch.Tensor, idx: int) -> torch.Tensor:
    return data[batch_index + idx].to(dtype=torch.long, non_blocking=True)


class Dataset:
    def __init__(self, ctx: Context, queue: multiprocessing.Queue, length: int):
        self.ctx = ctx
        self.length = length
        self.queue = queue

    def __len__(self):
        return self.length

    def __iter__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        yield next(self)

    def __next__(self):
        items = [self.queue.get() for _ in range(self.ctx.optimizer.gradient_accumulation_steps)]
        return torch.stack([itm for itm in items], 0)


def _process_fn(ctx: Context, queue: multiprocessing.Queue, idx: int, worker_count: int):
    data = torch.load(ctx.dataset.file_name)
    data_len = data.size(0) // worker_count
    data = data[data_len * idx:data_len * (idx + 1)]
    batch_index = torch.arange(0, ctx.model.batch_size).view(-1, 1)
    item_index = torch.arange(0, ctx.model.sequence_length).view(1, -1)
    batch_index = batch_index + item_index
    length = data.size(0) - ctx.model.batch_size * ctx.model.sequence_length

    random.seed(idx)
    while True:
        queue.put(get_sample(data, batch_index, random.randint(0, length)))


def get_dataset(ctx: Context) -> Dataset:
    if ctx.dataset.prefetch_factor < ctx.dataset.num_workers:
        print(f"Warning: prefetch_factor ({ctx.dataset.prefetch_factor}) < num_workers ({ctx.dataset.num_workers})."
              f"Some workers will be idle at all times. Reducing num_workers ({ctx.dataset.num_workers}) to "
              f"prefetch_factor ({ctx.dataset.prefetch_factor}).")
    queue = multiprocessing.Queue(ctx.dataset.prefetch_factor)
    workers = min(ctx.dataset.num_workers, ctx.dataset.prefetch_factor)
    procs = [multiprocessing.Process(target=_process_fn, args=(ctx, queue, idx, workers), daemon=True) for idx in
             range(workers)]
    for p in procs:
        p.start()
    data = torch.load(ctx.dataset.file_name)
    return Dataset(ctx, queue, data.size(0))
