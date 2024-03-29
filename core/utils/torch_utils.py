import os
import numpy as np
import torch


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return x

def to_np(t):
    return t.cpu().detach().numpy()

def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
