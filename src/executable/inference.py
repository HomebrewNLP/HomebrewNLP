import torch

from src.dataclass import Context
from src.model import LinearAttention
from src.utils.setup import encode, decode, get_model


def complete(ctx: Context, model: LinearAttention, prompt: str, temperature: float, generated_tokens: int) -> str:
    out = inp = encode(prompt).to(ctx.model.device).view(1, -1)
    for i in range(len(prompt), len(prompt) + generated_tokens):
        tmp = model(inp)[:, :, -1]
        tmp += torch.rand_like(tmp).clamp(min=1e-9).log().neg().log() * (-temperature)
        new_item = torch.argmax(tmp, -1).view(1, -1)
        out = inp = torch.cat([out, new_item], -1)
        if ctx.eval.cache:
            inp = new_item
    model.reset_cache()
    return decode(out)


@torch.no_grad()
def inference_cli(ctx: Context, temperature: float, generated_tokens: int):
    mod = get_model(ctx, False)
    mod.eval()
    while True:
        try:
            prompt = input("Prompt: ")
        except KeyboardInterrupt:
            break
        print(complete(ctx, mod, prompt, temperature, generated_tokens))
