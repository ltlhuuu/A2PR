"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 下午4:13
# @Author  : ltl
# @FileName: distancenet.py
# @Software: PyCharm
# @Github ：https://github.com/ltlhuuu
===========================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from rlkit.torch.networks import Mlp, CNN
class DistanceNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DistanceNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        distance = F.relu(self.l1(torch.cat([state, action], 1)))
        distance = F.relu(self.l2(distance))
        distance = self.l3(distance)
        return distance
