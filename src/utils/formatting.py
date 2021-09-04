import time
import typing
from typing import Optional

import torch
import wandb
from rich import print as rich_print
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
    rich_print(*data)


def log(*data, log_locals: bool = False):
    console.log(*data, log_locals=log_locals)


class WandbLog:
    def __init__(self, ctx: Context, steps: int):
        self.mean_loss = 0
        self.start_time = time.time()
        self.ctx = ctx
        self.idx = 0
        self.prev = 0
        self.steps = steps

    def __call__(self, current_loss: torch.Tensor, learning_rate: float, betas: typing.Tuple[float, float]):
        curr_loss = current_loss.item() / self.ctx.log.loss_steps_per_print
        self.idx += 1
        self.mean_loss = (self.mean_loss * self.prev + curr_loss * self.idx) / (self.prev + self.idx)  # LWMA
        self.prev += self.idx

        rate = self.ctx.log.loss_steps_per_print * self.idx / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * self.ctx.model.batch_size * self.ctx.model.sequence_length

        pretty_print(f"[{self.idx * self.ctx.log.loss_steps_per_print:{len(str(self.steps))}d}/{self.steps}]",
                     f"Loss: {curr_loss:7.4f} -",
                     f"Mean: {self.mean_loss:7.4f} |",
                     f"LR: {learning_rate:.6f} -",
                     f"Beta1: {betas[0]:.3f} -",
                     f"Beta2: {betas[1]:.3f} |",
                     f"Batch/s: {rate:6.3f} -",
                     f"Tokens/day: {tokens_per_day:11,.0f}")
        wandb.log({"Loss/Current": curr_loss,
                   "Loss/Mean": self.mean_loss,
                   "Speed/Batches per Second": rate,
                   "Speed/Tokens per Day": tokens_per_day,
                   "Optimizer/Learning Rate": learning_rate,
                   "Optimizer/Beta 1": betas[0],
                   "Optimizer/Beta 2": betas[1]},
                  step=self.idx * self.ctx.log.loss_steps_per_print)
