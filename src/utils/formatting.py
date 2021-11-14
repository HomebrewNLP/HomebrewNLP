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
        self.mean_max_loss = 0
        self.mean_acc = 0
        self.mean_max_acc = 0
        self.start_time = time.time()
        self.ctx = ctx
        self.idx = 0
        self.prev = 0
        self.steps = steps

    def normalize(self, var: torch.Tensor, attribute: str) -> float:
        attr = getattr(self, attribute)
        curr_var = var.item() / self.ctx.log.loss_steps_per_print / self.ctx.optimizer.gradient_accumulation_steps
        setattr(self, attribute, (attr * self.prev + curr_var * self.idx) / (self.prev + self.idx))  # LWMA
        return curr_var

    def __call__(self, current_loss: torch.Tensor, max_loss: torch.Tensor,
                 current_acc: torch.Tensor, max_acc: torch.Tensor, learning_rate: float,
                 betas: typing.Tuple[float, float]):
        self.idx += 1
        current_loss = self.normalize(current_loss, "mean_loss")
        current_acc = self.normalize(current_acc, "mean_acc")
        if self.ctx.optimizer.sharpness_aware_minimization:
            max_loss = self.normalize(max_loss, "mean_max_loss")
            max_acc = self.normalize(max_acc, "mean_max_acc")
        else:
            max_loss = max_acc = self.mean_max_loss = self.mean_max_acc = None
        self.prev += self.idx

        rate = self.ctx.log.loss_steps_per_print * self.idx / (time.time() - self.start_time)
        tokens_per_day = 3600 * 24 * rate * self.ctx.model.batch_size * self.ctx.model.sequence_length
        tokens_per_day *= self.ctx.optimizer.gradient_accumulation_steps

        pretty_print(f"[{self.idx * self.ctx.log.loss_steps_per_print:{len(str(self.steps))}d}/{self.steps}]",
                     f"Loss: {current_loss:7.4f} -",
                     f"Mean: {self.mean_loss:7.4f} |",
                     f"Acc: {current_acc:7.4f} -",
                     f"Mean: {self.mean_acc:7.4f} |",
                     f"LR: {learning_rate:.6f} -",
                     f"Beta1: {betas[0]:.3f} -",
                     f"Beta2: {betas[1]:.3f} |",
                     f"Batch/s: {rate:6.3f} -",
                     f"Tokens/day: {tokens_per_day:11,.0f}")

        wandb.log({"Loss/Current": current_loss,
                   "Loss/Mean": self.mean_loss,
                   "Loss/Current Max": max_loss,
                   "Loss/Mean Max": self.mean_max_loss,
                   "Accuracy/Current": current_acc,
                   "Accuracy/Mean": self.mean_acc,
                   "Accuracy/Current Max": max_acc,
                   "Accuracy/Mean Max": self.mean_max_acc,
                   "Speed/Batches per Second": rate,
                   "Speed/Tokens per Day": tokens_per_day,
                   "Optimizer/Learning Rate": learning_rate,
                   "Optimizer/Beta1": betas[0],
                   "Optimizer/Beta2": betas[1]},
                  step=self.idx * self.ctx.log.loss_steps_per_print)
