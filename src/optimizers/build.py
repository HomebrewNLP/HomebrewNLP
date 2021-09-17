import inspect
import traceback
import typing

import torch

from src.dataclass import Context
from src.optimizers import shampoo
from src.utils.formatting import pretty_print

OPTIMIZERS = {'Shampoo': shampoo.Shampoo}


def build_optimizer(ctx: Context, parameters: typing.Iterable[torch.nn.Parameter]):
    if ctx.optimizer.type in OPTIMIZERS:
        return OPTIMIZERS[ctx.optimizer.type](parameters, ctx.optimizer)
    optm = getattr(torch.optim, ctx.optimizer.type)
    if torch.optim.Optimizer not in inspect.getmro(optm):
        raise ValueError("Optimizer must inherit from 'torch.optim.Optimizer'.")
    params = {'params': parameters}
    for key in inspect.signature(optm).parameters.keys():
        if key in ctx.optimizer.serialize():
            params[key] = getattr(ctx.optimizer, key)
    return optm(**params)
