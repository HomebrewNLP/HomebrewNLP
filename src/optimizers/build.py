import copy
import typing
import inspect
import torch
import traceback

from . import shampoo
from src.utils.formatting import log, pretty_print
from src.dataclass import Context

OPTIMIZERS = {
    'AdamW': torch.optim.AdamW,
    'Shampoo': shampoo.Shampoo
}


def build_optimizer(ctx: Context, parameters: typing.List[torch.nn.Parameter]):
    name = ctx.optimizer.type
    optm = OPTIMIZERS.get(name)
    if name in ['AdamW']:
        return optm(params=parameters, weight_decay=ctx.optimizer.weight_decay)
    if name in ['Shampoo']:
        return optm(parameters, ctx=ctx.optimizer)
    try:
        optm = getattr(torch.optim, name)
        if not torch.optim.Optimizer in inspect.getmro(optm):
            raise ValueError("Optimizer must inherit from 'torch.optim.Optimizer'.")
        params = {'params': parameters}
        for key in inspect.signature(optm).parameters.keys():
            if key in ctx.optimizer:
                params[key] = getattr(ctx.optimizer, key)
        return optm(**params)
    except:
        pretty_print(f'{name} is not a valid optimizer type.')
        log(f'{name} is not a valid optimizer type.')
        traceback.print_exc()
