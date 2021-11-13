import multiprocessing
import random
import typing

import torch
import torch.utils.data

from src.dataclass import Context


@torch.jit.script
def get_sample(data: torch.Tensor, batch_index: torch.Tensor, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    dat = data[batch_index + idx]
    dat = dat.to(dtype=torch.long, non_blocking=True)
    return dat[:, :-1], dat[:, 1:]


class Dataset:
    def __init__(self, ctx: Context, length: int, queue: multiprocessing.Queue):
        self.length = length
        self.batch = ctx.optimizer.gradient_accumulation_steps
        self.queue = queue

    def __len__(self):
        return self.length

    def __iter__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        yield next(self)

    def __next__(self):
        return self.queue.get()


def _get_process_fn(ctx: Context, queue: multiprocessing.Queue) -> typing.Tuple[typing.Callable[[int], None], int]:
    data = torch.load(ctx.dataset.file_name)
    batch_index = torch.arange(0, ctx.model.batch_size).view(-1, 1)
    item_index = torch.arange(0, ctx.model.sequence_length + 1).view(1, -1)
    batch_index = batch_index + item_index
    length = data.size(0) - ctx.model.batch_size * ctx.model.sequence_length

    def _fn(idx):
        random.seed(idx)
        while True:
            queue.put(get_sample(data, batch_index, random.randint(0, length)))

    return _fn, length


def get_dataset(ctx: Context) -> Dataset:
    if ctx.dataset.prefetch_factor < ctx.dataset.num_workers:
        print(f"Warning: prefetch_factor ({ctx.dataset.prefetch_factor}) < num_workers ({ctx.dataset.num_workers})."
              f"Some workers will be idle at all times. Reducing num_workers ({ctx.dataset.num_workers}) to "
              f"prefetch_factor ({ctx.dataset.prefetch_factor}).")
    queue = multiprocessing.Queue(ctx.dataset.prefetch_factor)
    proc_fn, length = _get_process_fn(ctx, queue)
    procs = [multiprocessing.Process(target=proc_fn, args=(idx,), daemon=True)
             for idx in range(min(ctx.dataset.num_workers, ctx.dataset.prefetch_factor))]
    for p in procs:
        p.start()
    return Dataset(ctx, length, queue)
