import typing

import torch

from src.dataclass import Context
from src.model import LinearAttention
from src.utils.setup import encode, decode, get_model


def complete_batch(ctx: Context, model: LinearAttention, prompt: torch.Tensor, temperature: float,
                   generated_tokens: int) -> typing.List[str]:
    batch, prompt_size = prompt.size()
    out = prompt
    for _ in range(prompt_size, prompt_size + generated_tokens):
        tmp = model(prompt)[:, :, -1]
        tmp += torch.rand_like(tmp).clamp(min=1e-9).log().neg().log() * (-temperature)
        new_item = torch.argmax(tmp, -1).view(batch, -1)
        out = prompt = torch.cat([out, new_item], -1)
        if ctx.eval.cache:
            prompt = new_item
    model.reset_cache()
    return [decode(o) for o in out.unbind(0)]


def complete(ctx: Context, model: LinearAttention, prompt: str, temperature: float, generated_tokens: int) -> str:
    return complete_batch(ctx, model, encode(prompt).to(dtype=torch.long, device=ctx.model.device).view(1, -1),
                          temperature, generated_tokens)[0]


@torch.no_grad()
def inference_cli(ctx: Context, temperature: float, generated_tokens: int):
    mod = get_model(ctx, True).model
    mod.eval()
    while True:
        try:
            prompt = input("Prompt: ")
        except KeyboardInterrupt:
            break
        print(complete(ctx, mod, prompt, temperature, generated_tokens))
