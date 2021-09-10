import inspect
import traceback
import typing

import torch

from src.dataclass import Context
from src.optimizers import shampoo
from src.utils.formatting import pretty_print

OPTIMIZERS = {'shampoo': shampoo.Shampoo}


def build_optimizer(ctx: Context, parameters: typing.Iterable[torch.nn.Parameter]):
    if name in OPTIMIZERS:
        return OPTIMIZERS[name](parameters, ctx.optimizer)
    try:
        optm = getattr(torch.optim, name)
        if torch.optim.Optimizer not in inspect.getmro(optm):
            raise ValueError("Optimizer must inherit from 'torch.optim.Optimizer'.")
        params = {'params': parameters}
        for key in inspect.signature(optm).parameters.keys():
            if key in ctx.optimizer:
                params[key] = getattr(ctx.optimizer, key)
        return optm(**params)
    except TypeError:
        pretty_print(f'{name} is not a valid optimizer type.')
        traceback.print_exc()
