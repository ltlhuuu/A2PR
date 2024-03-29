from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.network import network_utils

class FCBody(nn.Module):
    def __init__(self, device, input_dim, hidden_units=(64, 64), activation=functional.relu, init_type='xavier', info=None):
        super().__init__()
        self.to(device)
        self.device = device
        dims = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([network_utils.layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        if init_type == "xavier":
            self.layers = nn.ModuleList([network_utils.layer_init_xavier(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        elif init_type == "uniform":
            self.layers = nn.ModuleList([network_utils.layer_init_uniform(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        elif init_type == "zeros":
            self.layers = nn.ModuleList([network_utils.layer_init_zero(nn.Linear(dim_in, dim_out).to(device)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        elif init_type == "constant":
            self.layers = nn.ModuleList([network_utils.layer_init_constant(nn.Linear(dim_in, dim_out).to(device), const=info) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.activation = activation
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    def compute_lipschitz_upper(self):
        return [np.linalg.norm(layer.weight.detach().cpu().numpy(), ord=2) for layer in self.layers]


class ConvBody(nn.Module):
    def __init__(self, device, state_dim, architecture):
        super().__init__()

        def size(size, kernel_size=3, stride=1, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        spatial_length, _, in_channels = state_dim
        num_units = None
        layers = nn.ModuleList()
        for layer_cfg in architecture['conv_layers']:
            layers.append(nn.Conv2d(layer_cfg["in"], layer_cfg["out"], layer_cfg["kernel"],
                                         layer_cfg["stride"], layer_cfg["pad"]))
            if not num_units:
                num_units = size(spatial_length, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
            else:
                num_units = size(num_units, layer_cfg["kernel"], layer_cfg["stride"], layer_cfg["pad"])
        num_units = num_units ** 2 * architecture["conv_layers"][-1]["out"]

        self.feature_dim = num_units
        self.spatial_length = spatial_length
        self.in_channels = in_channels
        self.layers = layers
        self.to(device)
        self.device = device

    def forward(self, x):
        x = functional.relu(self.layers[0](self.shape_image(x)))
        for idx, layer in enumerate(self.layers[1:]):
            x = functional.relu(layer(x))
        # return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)
