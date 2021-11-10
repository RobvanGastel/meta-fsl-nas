import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def create_one_hot(act, act_dim):
    index = torch.eye(act_dim).cuda()
    return index[act.long()]


"""GRU Actor Critic implementation for discrete action space"""


class GRUCategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.act_dim = act_dim

        self.Linear1 = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size+act_dim+1,  # +1 for the reward
                          hidden_size,
                          batch_first=True)
        self.Linear2 = nn.Linear(hidden_size, act_dim)

    def act(self, obs, prev_act, prev_rew, hid):

        action_logits, hidden_out = self.forward(
            obs, prev_act, prev_rew, hid)

        # Greedy action selection
        greedy_actions = torch.argmax(action_logits, dim=-1)

        return greedy_actions, hidden_out

    def sample(self, obs, prev_act, prev_rew, hid):

        # print("obs", obs)
        action_logits, hidden_out = self.forward(
            obs, prev_act, prev_rew, hid)

        if torch.any(torch.isnan(obs)):
            print("obs:", obs)
        # print(action_logits)
        action_probs = F.softmax(action_logits, dim=-1)

        if torch.any(torch.isnan(action_probs)):
            print("action_probs:", action_probs)
        # print(action_probs)
        action_dist = Categorical(action_probs)

        # if torch.any(torch.isnan(action_dist)):
        #     print("action_dist:", action_dist)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, hidden_out

    def forward(self, obs, prev_act, prev_rew, hid):

        self.gru.flatten_parameters()

        prev_act = create_one_hot(prev_act, self.act_dim)
        gru_input = self.activation(self.Linear1(obs))
        gru_input = torch.cat(
            [
                gru_input,
                prev_act,
                prev_rew
            ],
            dim=2,
        )

        # unroll the GRU network
        gru_out, hid_out = self.gru(gru_input.float(), hid.float())
        print(gru_out.shape)
        q = self.Linear2(gru_out)

        return q, hid_out


class GRUQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.act_dim = act_dim

        self.Linear1 = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size+act_dim+1,  # +1 for the reward
                          hidden_size,
                          batch_first=True)
        self.Linear2 = nn.Linear(hidden_size, act_dim)

    def forward(self, obs, prev_act, prev_rew, hid):

        self.gru.flatten_parameters()

        prev_act = create_one_hot(prev_act, self.act_dim)
        gru_input = self.activation(self.Linear1(obs))
        gru_input = torch.cat(
            [
                gru_input,
                prev_act,
                prev_rew
            ],
            dim=2,
        )

        # unroll the GRU network
        gru_out, hid_out = self.gru(gru_input.float(), hid.float())
        q = self.Linear2(gru_out)

        return q, hid_out


class GRUActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_size=[256, 256],
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        self.pi = GRUCategoricalPolicy(
            obs_dim, act_dim, hidden_size[0], activation)
        self.q1 = GRUQNetwork(obs_dim, act_dim, hidden_size[0], activation)
        self.q2 = GRUQNetwork(obs_dim, act_dim, hidden_size[0], activation)

    def explore(self, obs, prev_a, prev_r, hid):
        # action selection using softmax
        with torch.no_grad():
            action, _, _, hid = self.pi.sample(obs, prev_a, prev_r, hid)
        return action.item(), hid

    def act(self, obs, prev_a, prev_r, hid):
        # Greedy action selection by the policy, argmax
        with torch.no_grad():
            action, hid = self.pi.act(obs, prev_a, prev_r, hid)
        return action.item(), hid


"""MLP Actor Critic implementation for discrete action space"""


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        pi_sizes = [obs_dim] + hidden_size + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh).cuda()

    def act(self, obs):
        action_logits = self.pi(obs)
        greedy_actions = torch.argmax(action_logits, dim=-1)
        return greedy_actions

    def sample(self, obs):
        policy = self.pi(obs)
        action_probs = F.softmax(policy, dim=-1)

        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        self.a = mlp([obs_dim] + hidden_size + [act_dim], activation)
        self.v = mlp([obs_dim] + hidden_size + [1], activation)

    def forward(self, obs):

        a = self.a(obs)
        v = self.v(obs)

        return v + a - a.mean(1, keepdim=True)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_size=[256, 256],
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        self.pi = CategoricalPolicy(obs_dim, act_dim, hidden_size, activation)
        self.q1 = MLPQNetwork(obs_dim, act_dim, hidden_size, activation)
        self.q2 = MLPQNetwork(obs_dim, act_dim, hidden_size, activation)

    def explore(self, obs):
        with torch.no_grad():
            action, _, _ = self.pi.sample(
                torch.as_tensor(obs, dtype=torch.float32).cuda())
        return action.item()

    def act(self, obs):
        # Greedy action selection by the policy
        with torch.no_grad():
            action = self.pi.act(
                torch.as_tensor(obs, dtype=torch.float32).cuda())
        return action.item()
