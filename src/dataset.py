import typing

import torch
import torch.utils.data

from src.dataclass import Context


@torch.jit.script
def get_sample(data: torch.Tensor, batch_index: torch.Tensor, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    dat = data[batch_index + idx]
    dat = dat.to(dtype=torch.long, non_blocking=True)
    return dat[:, :-1], dat[:, 1:]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ctx: Context, workers: int):
        self.ctx = ctx
        self.workers = workers
        self.data = torch.empty((1,))
        data = torch.load(self.ctx.dataset.file_name)
        self.length = data.size(0) - self.ctx.model.batch_size * self.ctx.model.sequence_length

        batch_index = torch.arange(0, ctx.model.batch_size).view(-1, 1)
        item_index = torch.arange(0, ctx.model.sequence_length + 1).view(1, -1)
        self.batch_index = batch_index + item_index
        self.worker_id: int = 0
        self.slice_size = self.length // self.workers

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return get_sample(self.data, self.batch_index, idx % self.slice_size)

    def set_id(self, worker_id: int):
        self.worker_id = worker_id
        data = torch.load(self.dataset.file_name)
        self.data = data[self.slice_size * worker_id: self.slice_size * (worker_id + 1)]


def get_dataset(ctx: Context) -> torch.utils.data.DataLoader:
    if ctx.dataset.prefetch_factor < ctx.dataset.num_workers:
        print(f"Warning: prefetch_factor ({ctx.dataset.prefetch_factor}) < num_workers ({ctx.dataset.num_workers})."
              f"Some workers will be idle at all times. Reducing num_workers ({ctx.dataset.num_workers}) to "
              f"prefetch_factor ({ctx.dataset.prefetch_factor}).")
    workers = min(ctx.dataset.num_workers, ctx.dataset.prefetch_factor)
    dset = Dataset(ctx, workers)
    return torch.utils.data.DataLoader(dset, ctx.optimizer.gradient_accumulation_steps, True,
                                       num_workers=workers, pin_memory=ctx.dataset.pin_memory,
                                       prefetch_factor=ctx.dataset.prefetch_factor,
                                       worker_init_fn=dset.set_id)
