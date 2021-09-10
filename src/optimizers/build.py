import inspect
import traceback
import typing

import torch

from src.dataclass import Context
from src.optimizers import shampoo
from src.utils.formatting import pretty_print

OPTIMIZERS = {'shampoo': shampoo.Shampoo}


def build_optimizer(ctx: Context, parameters: typing.Iterable[torch.nn.Parameter]):
    name = ctx.optimizer.type
    if name in OPTIMIZERS:
        return OPTIMIZERS[name](parameters, ctx.optimizer)
    optm = getattr(torch.optim, name)
    if torch.optim.Optimizer not in inspect.getmro(optm):
        raise ValueError("Optimizer must inherit from 'torch.optim.Optimizer'.")
    params = {'params': parameters}
    for key in inspect.signature(optm).parameters.keys():
        if hasattr(ctx.optimizer,key):
            params[key] = getattr(ctx.optimizer, key)
    optm = optm(**params)
    return optm
