import numpy as np
import torch

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

def arcradians(cos, sin):
    if cos > 0 and sin > 0:
        return np.arccos(cos)
    elif cos > 0 and sin < 0:
        return np.arcsin(sin)
    elif cos < 0 and sin > 0:
        return np.arccos(cos)
    elif cos < 0 and sin < 0:
        return -1 * np.arccos(cos)


def normalize_rows(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def copy_row(x, num_rows):
    return np.multiply(np.ones((num_rows, 1)), x)

def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

def search_same_row(matrix, target_row):
    idx = np.where(np.all(matrix == target_row, axis=1))
    return idx