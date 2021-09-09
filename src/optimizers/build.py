import copy
import typing
import inspect
import torch
import traceback

from . import shampoo
from src.utils.formatting import log, pretty_print
from src.dataclass import Context

OPTIMIZERS = {
    'AdamW':torch.optim.AdamW,
    'Shampoo': shampoo.Shampoo
}

def build_optimizer(ctx: Context, parameters: typing.List[torch.nn.Parameter]):
    name = ctx.optimizer.type
    optm = OPTIMIZERS.get(name,None)
    if name in ['AdamW']:
        return optm(params=parameters, weight_decay=ctx.optimizer.weight_decay)
    elif name in ['Shampoo']:
        return optm(parameters, ctx=ctx.optimizer)
    else:
        try:
            optm = getattr(torch.optim,name)
            assert torch.optim.Optimizer in inspect.getmro(optm)
            params = {'params':parameters}
            for key, value in inspect.signature(optm).parameters.items():
                if key in ctx.optimizer:
                    params[key]=getattr(ctx.optimizer,key)
            return optm(**params)
        except Exception as e:
            pretty_print(f'{name} is not a valid optimizer type.')
            log(f'{name} is not a valid optimizer type.')
            traceback.print_exc()