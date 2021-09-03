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
        self.mean_loss = 0
        self.start_time = time.time()
        self.ctx = ctx
        self.idx = 0
        self.steps = steps

    def __call__(self, current_loss: torch.Tensor, learning_rate: float):
        curr_loss = current_loss.item() / self.ctx.log.loss_steps_per_print
        del current_loss

        self.mean_loss = (self.mean_loss * self.idx + curr_loss * (self.idx + 1)) / (self.idx * 2 + 1)
        self.idx += 1

        rate = self.ctx.log.loss_steps_per_print * self.idx / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * self.ctx.model.batch_size * self.ctx.model.sequence_length

        pretty_print(f"[{self.idx * self.ctx.log.loss_steps_per_print:{len(str(self.steps))}d}/{self.steps}]",
                     f"Loss: {curr_loss:7.4f} -",
                     f"Mean: {self.mean_loss:7.4f} |",
                     f"LR: {learning_rate:.6f} |",
                     f"Batch/s: {rate:6.3f} -",
                     f"Tokens/day: {tokens_per_day:11,.0f}")
        wandb.log({"Loss": curr_loss,
                   "Mean Loss": self.mean_loss,
                   "Learning Rate": learning_rate,
                   "Batches per Second": rate,
                   "Tokens per Day": tokens_per_day})
