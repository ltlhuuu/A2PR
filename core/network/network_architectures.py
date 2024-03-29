import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from core.network import network_utils, network_bodies
from core.utils import torch_utils


class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x:x):
        super().__init__()
        body = network_bodies.FCBody(device, input_units, hidden_units=tuple(hidden_units), init_type='xavier')
        self.body = body
        self.fc_head = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, output_units, bias=True), bias=True)
        self.device = device
        self.head_activation = head_activation
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.body(x)
        y = self.fc_head(y)
        y = self.head_activation(y)
        return y

class DoubleCriticDiscrete(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units):
        super().__init__()
        self.device = device
        self.q1_net = FCNetwork(device, input_units, hidden_units, output_units)
        self.q2_net = FCNetwork(device, input_units, hidden_units, output_units)
    
    # def forward(self, x, a):
    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        recover_size = False
        if len(x.size()) == 1:
            recover_size = True
            x = x.reshape((1, -1))
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        if recover_size:
            q1 = q1[0]
            q2 = q2[0]
        return q1, q2


class DoubleCriticNetwork(nn.Module):
    def __init__(self, device, num_inputs, num_actions, hidden_units):
        super(DoubleCriticNetwork, self).__init__()
        self.device = device

        # Q1 architecture
        self.body1 = network_bodies.FCBody(device, num_inputs + num_actions, hidden_units=tuple(hidden_units))
        self.head1 = network_utils.layer_init_xavier(nn.Linear(self.body1.feature_dim, 1))
        # Q2 architecture
        self.body2 = network_bodies.FCBody(device, num_inputs + num_actions, hidden_units=tuple(hidden_units))
        self.head2 = network_utils.layer_init_xavier(nn.Linear(self.body2.feature_dim, 1))

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor): state = torch_utils.tensor(state, self.device)
        recover_size = False
        if len(state.shape) > 2:
            state = state.view(state.shape[0], -1)
            action = action.view(action.shape[0], -1)
        elif len(state.shape) == 1:
            state = state.view(1, -1)
            action = action.view(1, -1)
            recover_size = True
        if not isinstance(action, torch.Tensor): action = torch_utils.tensor(action, self.device)

        xu = torch.cat([state, action], 1)

        q1 = self.head1(self.body1(xu))
        q2 = self.head2(self.body2(xu))
        
        if recover_size:
            q1 = q1[0]
            q2 = q2[0]
        return q1, q2
    
