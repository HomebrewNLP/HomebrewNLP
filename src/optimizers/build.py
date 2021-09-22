import inspect
import typing

import deepspeed.ops.adam
import torch

from src.dataclass import Context
from src.optimizers import shampoo

OWN_OPTIMIZER = {'Shampoo': shampoo.Shampoo}
LIB_OPTIMIZER = {'DeepSpeedCPUAdam': deepspeed.ops.adam.DeepSpeedCPUAdam}


def build_optimizer(ctx: Context, parameters: typing.Iterable[torch.nn.Parameter]):
    opt_type = ctx.optimizer.type
    if opt_type in OWN_OPTIMIZER:
        return OWN_OPTIMIZER[opt_type](parameters, ctx.optimizer)
    opt = LIB_OPTIMIZER[opt_type] if opt_type in LIB_OPTIMIZER else getattr(torch.optim, opt_type)
    if torch.optim.Optimizer not in inspect.getmro(opt):
        raise ValueError("Optimizer must inherit from 'torch.optim.Optimizer'.")
    params = {key: getattr(ctx.optimizer, key) for key in inspect.signature(opt).parameters.keys()
              if key in ctx.optimizer.serialize()}
    return opt(parameters, **params)
