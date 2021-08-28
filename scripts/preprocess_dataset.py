import numpy as np
import torch

with open("data.txt", 'rb') as f:
    dat = f.read()
tensor = torch.as_tensor(np.frombuffer(dat, np.uint8))
torch.save(tensor, '../out.tensor')
