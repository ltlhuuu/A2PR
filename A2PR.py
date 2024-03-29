import copy
import numpy as np
import torch
import torch.nn.functional as F
from core.network.actor import A2PR_Actor
from core.network.critic import Critic, Value
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

class A2PR(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, policy_noise=0.2,
                noise_clip=0.5, policy_freq=2, actor_lr=3e-4, critic_lr=3e-4, alpha=2.5, mask=1.0, vae_weight=1.0):
        self.device = device
        self.actor = A2PR_Actor(state_dim, action_dim, max_action).to(self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_dim,action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.value = Value(state_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        self.advantage_list = []
        self.total_it = 0
        # VAE
        latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        self.mask = mask
        self.vae_weight = vae_weight

        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "value": self.value,
            "value_optimizer": self.value_optimizer,
            "vae": self.vae,
            "vae_optimizer": self.vae_optimizer,
        }

        print("state_dim:", state_dim, "action_dim:", action_dim)

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1
        tb_statics = dict()

        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        # Compute the target part so we should use torch.no_grad() to clear the gradient
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            next_v = self.value(next_state)

        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        tb_statics.update({"critic_loss": critic_loss.item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            next_current_Q1, next_current_Q2 = self.critic(next_state, next_action)
            next_Q = torch.min(next_current_Q1, next_current_Q2)
            targ_Q = reward + not_done * self.discount * next_Q
            t_Q = torch.min(targ_Q, target_Q)
        value = self.value(state)
        value_loss = F.mse_loss(value, t_Q)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        recon, mean, std = self.vae(state, action)
        Q_vae = self.critic.Q1(state, action)

        embedding = (Q_vae > value)
        weight = self.vae_weight
        weight_emb = weight * embedding
        recon_loss = (weight_emb * torch.square(recon - action)).mean()
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        tb_statics.update({"vae_loss": vae_loss.item(),
                           "recon_loss": recon_loss.item(),
                           "KL_loss": KL_loss.item(),
                           })

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean()
            recon_pi, mean, std = self.vae(state, pi)
            Q_new = self.critic.Q1(state, action)
            advantage_action = Q_new - value
            Q_pi = self.critic.Q1(state, recon_pi)
            advantage_pi = Q_pi - value
            embedding_pi = recon_pi * (advantage_pi >= advantage_action)
            embedding_action = action * (advantage_action > advantage_pi)
            embedding = embedding_pi + embedding_action
            adv_pi = (Q_pi.detach() >= value.detach())
            adv_action = (Q_new.detach() > value.detach())
            adv = self.mask * (adv_pi + adv_action)
            bc_loss = (torch.square(pi - embedding) * adv).mean()

            # Optimize the actor
            combined_loss = actor_loss + bc_loss
            self.actor_optimizer.zero_grad()
            combined_loss.backward()
            self.actor_optimizer.step()

            tb_statics.update(
                {
                    "bc_loss": bc_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "combined_loss": combined_loss.item(),
                    "Q_value": torch.mean(Q).item(),
                    "targ_Q": torch.mean(targ_Q).item(),
                    "target_Q": torch.mean(target_Q).item(),
                    "lmbda": lmbda,

                }
            )

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1-self.tau) * target_param.data
                )
        return tb_statics

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_list = np.array(advantage_list)
        return torch.tensor(advantage_list, dtype=torch.float)

    def save(self, model_path):
        state_dict = dict()
        for model_name, model in self.models.items():
            state_dict[model_name] = model.state_dict()
        torch.save(state_dict, model_path)

    def load(self,model_path):
        state_dict = torch.load(model_path)
        for model_name, model in self.models.items():
            model.load_state_dict(state_dict[model_name])

    def add_data(self, new_data):
        if len(self.advantage_list) >= 10:
            self.advantage_list.pop(0)
        self.advantage_list.append(new_data)
