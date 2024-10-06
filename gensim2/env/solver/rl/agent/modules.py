import torch.nn as nn
import torch.nn.functional as F
import torch
import utils


class DiscreteCritic(nn.Module):
    def __init__(self, repr_dim, n_actions, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            #    nn.Linear(feature_dim, feature_dim),nn.ReLU()
        )

        self.Vnet = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Anet = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        v = self.Vnet(h)
        a = self.Anet(h)
        q = v + a - a.mean(1, keepdim=True)

        return q


class DiscreteActor(nn.Module):
    def __init__(self, repr_dim, n_actions, feature_dim, hidden_dim, critic=None):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

        self.critic = critic
        self.apply(utils.weight_init)

    def forward(self, obs, return_action=False, *args, **kwargs):
        if self.critic is None:
            h = self.trunk(obs)
            actions = self.policy(h)
            # dist = F.gumbel_softmax(actions, tau=1, hard=False)
        else:
            actions = self.critic(obs)

        dist = utils.MultiNomial(actions)

        if return_action:
            return actions

        return dist


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
