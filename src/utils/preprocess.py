import ftfy
import torch
from typing import Optional

from src.utils.utils import encode


def preprocess_data(in_path: str, out_path: str):
    # Todo: convert to pathlib and confirm paths existance
    with open(in_path, 'r', errors="ignore") as f:
        dat = f.read()
    dat = ftfy.fix_text(dat)
    torch.save(encode(dat), out_path)
