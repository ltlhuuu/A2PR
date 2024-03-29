import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from core.network import network_utils, network_bodies
from core.utils import torch_utils


class MLPCont(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, action_range=1.0, init_type='xavier'):
        super().__init__()
        self.device = device
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.action_range = action_range

    """https://github.com/hari-sikchi/AWAC/blob/3ad931ec73101798ffe82c62b19313a8607e4f1e/core.py#L91"""
    def forward(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        recover_size = False
        if len(obs.size()) == 1:
            recover_size = True
            obs = obs.reshape((1, -1))
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range

        log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)

        if recover_size:
            pi_action, logp_pi = pi_action[0], logp_pi[0]
        return pi_action, logp_pi

    def get_logprob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
            self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        return logp_pi


class MLPDiscrete(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, init_type='xavier'):
        super().__init__()
        self.device = device
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
    
    def forward(self, obs, deterministic=True):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        recover_size = False
        if len(obs.size()) == 1:
            recover_size = True
            obs = obs.reshape((1, -1))
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        action = m.sample()
        logp = m.log_prob(action)
        if recover_size:
            action, logp = action[0], logp[0]
        return action, logp
    
    def get_logprob(self, obs, actions):
        if not isinstance(obs, torch.Tensor):
            obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch_utils.tensor(actions, self.device)
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        logp_pi = m.log_prob(actions)
        return logp_pi
    