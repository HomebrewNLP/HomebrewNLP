import torch

from src.dataclass import Context
from src.train import main as train


def main(ctx: Context, chrome_trace_path: str = "torch_trace", steps: int = 128):
    with torch.autograd.profiler.profile(use_cuda=True, use_cpu=False, use_kineto=True) as prof:
        train(ctx, steps)
    print(prof.key_averages())
    if chrome_trace_path:
        prof.export_chrome_trace(chrome_trace_path)
