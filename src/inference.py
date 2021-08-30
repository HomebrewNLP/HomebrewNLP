import torch

from src.utils import encode, decode


def complete(model: torch.nn.Module, prompt: str, temperature: float, generated_tokens: int) -> str:
    inp = encode(prompt)
    for i in range(len(prompt), len(prompt) + generated_tokens):
        new_item = torch.distributions.one_hot_categorical.OneHotCategorica(model(inp)[i] / temperature).sample()
        inp = torch.cat([inp, new_item], -1)
    return decode(inp)
