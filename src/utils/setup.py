import math
import random

import numpy as np
import torch

from src.dataclass import Context
from src.model import LinearAttention
from src.utils.formatting import pretty_print


# skipcq: PYL-W0212
def setup_torch(seed: int):
    torch._C._debug_set_autodiff_subgraph_inlining(False)
    torch._C._set_graph_executor_optimize(True)
    torch._C._set_backcompat_broadcast_warn(False)
    torch._C._set_backcompat_keepdim_warn(False)
    torch._C._set_cudnn_enabled(True)
    torch._C._set_mkldnn_enabled(True)
    torch._C._set_mkldnn_enabled(True)
    torch._C._set_cudnn_benchmark(True)
    torch._C._set_cudnn_deterministic(False)
    torch._C._set_cudnn_allow_tf32(True)
    torch._C._set_cublas_allow_tf32(True)
    torch._C._jit_set_inline_everything_mode(True)

    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_set_nvfuser_enabled(False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model(ctx: Context, load_model: bool) -> LinearAttention:
    mod = LinearAttention(ctx).to(dtype=torch.float16 if ctx.model.float16 else torch.float)

    if ctx.model.print_on_init:
        pretty_print(str(mod))

    parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, mod.parameters()))
    base = int(math.log10(parameters) / 3)
    pretty_print(f'Parameters: {parameters / (1000 ** base):.1f}{" kMBT"[base]}')
    if load_model:
        mod.load()
    if not ctx.model.offloading:
        mod = mod.to(ctx.model.device)
    return mod


def encode(prompt: str) -> torch.Tensor:
    return torch.as_tensor(np.frombuffer(prompt.encode('UTF-8'), np.uint8))


def decode(output: torch.LongTensor) -> str:
    return ''.join(chr(c) for c in output.view(-1).unbind(0))
