import torch.nn as nn


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def layer_init_zero(layer, bias=True):
    nn.init.constant_(layer.weight, 0)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_constant(layer, const, bias=True):
    nn.init.constant_(layer.weight, const)
    if bias:
        nn.init.constant_(layer.bias.data, const)
    return layer


def layer_init_xavier(layer, bias=True):
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_uniform(layer, low=-0.003, high=0.003, bias=0):
    nn.init.uniform_(layer.weight, low, high)
    if not (type(bias)==bool and bias==False):
        nn.init.constant_(layer.bias.data, bias)
    return layer
