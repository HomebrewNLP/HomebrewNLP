import time
from typing import Optional

import torch
import wandb
from rich import print
from rich.console import Console
from rich.syntax import Syntax

from src.dataclass import Context

# Color coded tracebacks
# install(show_locals=False, extra_lines=0)
console = Console()


# TODO: Allow for users to choose theme
def syntax_print(string: str, language: Optional[str] = "python", theme: Optional[str] = "monokai",
                 title: Optional[str] = None) -> None:
    if title is not None:
        console.rule(title)
    syntax = Syntax(string, language, theme=theme, line_numbers=True)
    console.print(syntax)


def pretty_print(*data):
    print(*data)


def log(*data, locals: bool = False):
    console.log(*data, log_locals=locals)


class WandbLog:
    def __init__(self, ctx: Context, steps: int):
        self.mean_loss = torch.zeros([], device=ctx.model.device,
                                     dtype=torch.float16 if ctx.model.float16 else torch.float)
        self.start_time = time.time()
        self.ctx = ctx
        self.idx = 0
        self.steps = steps

    def __call__(self, current_loss: torch.Tensor, learning_rate: float):
        self.idx += 1
        self.mean_loss += current_loss
        curr_loss = current_loss.item() / self.ctx.log.loss_steps_per_print
        rate = self.idx / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * self.ctx.model.batch_size * self.ctx.model.sequence_length
        mean_loss = self.mean_loss.item() / self.idx

        pretty_print(f"[{self.idx:{len(str(self.steps))}d}/{self.steps}]",
                     f"Loss: {curr_loss:7.4f} -",
                     f"Mean: {mean_loss:7.4f} |",
                     f"LR: {learning_rate:.6f} |",
                     f"Batch/s: {rate:6.3f} -",
                     f"Tokens/day: {tokens_per_day:11,.0f}")
        wandb.log({"Loss": current_loss,
                   "Mean Loss": mean_loss,
                   "Learning Rate": learning_rate,
                   "Batches/sec": rate,
                   "Tokens/Day": tokens_per_day})
