import torch

from src.dataclass import Context
from src.utils.setup import encode, decode


def complete(ctx: Context, model: torch.nn.Module, prompt: str, temperature: float, generated_tokens: int) -> str:
    out = inp = encode(prompt)

    for i in range(len(prompt), len(prompt) + generated_tokens):
        new_item = torch.distributions.one_hot_categorical.OneHotCategorical(model(inp)[i] / temperature).sample()
        out = torch.cat([out, new_item], -1)
        inp = new_item if ctx.eval.cache else out
    # TODO: Reset cache
    return decode(inp)
