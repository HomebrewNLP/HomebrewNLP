import torch

from src.dataclass import Context
from src.train import train_model


def main(ctx: Context, chrome_trace_path: str = "torch_trace", steps: int = 128):
    with torch.autograd.profiler.profile(use_cuda=True, use_cpu=False, use_kineto=True) as prof:
        train_model(ctx, steps)
    print(prof.key_averages())
    if chrome_trace_path:
        prof.export_chrome_trace(chrome_trace_path)
