import typing
import torch
import torch.utils.data

from src.utils import _xla
from src.dataclass import Context


@torch.jit.script
def get_sample(data: torch.Tensor, batch_index: torch.Tensor, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    dat = data[batch_index + idx]
    dat = dat.to(dtype=torch.long, non_blocking=True)
    return dat[:, :-1], dat[:, 1:]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ctx: Context):
        self.data = torch.load(ctx.dataset.file_name)
        batch_index = torch.arange(0, ctx.model.batch_size).view(-1, 1)
        item_index = torch.arange(0, ctx.model.sequence_length + 1).view(1, -1)
        self.batch_index = batch_index + item_index
        self.length = self.data.size(0) - ctx.model.batch_size * ctx.model.sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return get_sample(self.data, self.batch_index, idx)


def get_dataset(ctx: Context) -> torch.utils.data.DataLoader:
    if ctx.dataset.prefetch_factor < ctx.dataset.num_workers:
        print(f"Warning: prefetch_factor ({ctx.dataset.prefetch_factor}) < num_workers ({ctx.dataset.num_workers})."
              f"Some workers will be idle at all times. Reducing num_workers ({ctx.dataset.num_workers}) to "
              f"prefetch_factor ({ctx.dataset.prefetch_factor}).")
    if ctx.model.xla.use_xla:
        if not _xla.xm.is_master_ordinal():
            _xla.xm.rendezvous('load_once')
        dataset = Dataset(ctx)
        if _xla.xm.is_master_ordinal():
            _xla.xm.rendezvous('load_once')
        sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas = _xla.xm.xrt_world_size(),
                rank = _xla.xm.get_ordinal(), shuffle = ctx.dataset.shuffle)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=ctx.optimizer.gradient_accumulation_steps,
                                           sampler=sampler,
                                           num_workers=min(ctx.dataset.num_workers, ctx.dataset.prefetch_factor),
                                           prefetch_factor=ctx.dataset.prefetch_factor
                                           )
    else:
        return torch.utils.data.DataLoader(Dataset(ctx), batch_size=ctx.optimizer.gradient_accumulation_steps,
                                           num_workers=min(ctx.dataset.num_workers, ctx.dataset.prefetch_factor),
                                           pin_memory=ctx.dataset.pin_memory, shuffle = ctx.dataset.shuffle,
                                           prefetch_factor=ctx.dataset.prefetch_factor)
