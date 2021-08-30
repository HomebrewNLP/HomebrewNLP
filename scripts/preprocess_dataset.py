import ftfy
import numpy as np
import torch

with open("data.txt", 'r', errors="ignore") as f:
    dat = f.read()
dat = ftfy.fix_text(dat).encode('UTF-8')
tensor = torch.as_tensor(np.frombuffer(dat, np.uint8))
torch.save(tensor, '../out.tensor')
