import ftfy
import torch

from src.utils import encode

with open("data.txt", 'r', errors="ignore") as f:
    dat = f.read()
dat = ftfy.fix_text(dat)
torch.save(encode(dat), '../out.tensor')
