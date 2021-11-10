import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64,
                 activation=nn.ReLU()):
        super().__init__()

        self.activation = activation

        self.Linear1 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, act_dim)

    def act(self, memory_embedding):
        action_logits = self.policy(
            self.activation(self.Linear1(memory_embedding)))

        # Greedy action selection
        return torch.argmax(action_logits, dim=-1)

    def sample(self, memory_embedding):
        action_logits = self.policy(
            self.activation(self.Linear1(memory_embedding)))

        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64,
                 activation=nn.ReLU()):
        super().__init__()

        self.activation = activation

        self.Linear1 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, act_dim)

    def forward(self, memory_embedding):
        return self.q(self.activation(self.Linear1(memory_embedding)))


class Memory(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64,
                 activation=nn.ReLU()):
        super().__init__()

        self.act_dim = act_dim
        self.activation = activation

        self.obs_encoding = nn.Linear(obs_dim, hidden_size)
        # +1 for the reward
        self.gru = nn.GRU(hidden_size+act_dim+1,
                          hidden_size,
                          batch_first=True)

    def forward(self, obs, prev_act, prev_rew, hid_in):
        self.gru.flatten_parameters()

        act_encoding = self._create_one_hot(prev_act)
        obs_encoding = self.activation(self.obs_encoding(obs))

        gru_input = torch.cat(
            [
                obs_encoding,
                act_encoding,
                prev_rew
            ],
            dim=2,
        ).float()

        memory_embedding, hid_out = self.gru(
            gru_input, hid_in.float())

        return memory_embedding, hid_out

    def _create_one_hot(self, act):
        index = torch.eye(self.act_dim).cuda()
        return index[act.long()]


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=[256, 256], activation=nn.ReLU()):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # Move outside?
        self.memory = Memory(obs_dim, act_dim, hidden_sizes[0])

        self.pi = CategoricalPolicy(obs_dim, act_dim,
                                    hidden_size=hidden_sizes[0],
                                    activation=activation)

        self.q1 = QNetwork(
            obs_dim, act_dim, hidden_size=hidden_sizes[0],
            activation=activation)
        self.q2 = QNetwork(
            obs_dim, act_dim, hidden_size=hidden_sizes[0],
            activation=activation)

    def act(self, obs, prev_act, prev_rew, hid_in):
        with torch.no_grad():
            mem_emb, hid_out = self.memory(obs, prev_act, prev_rew, hid_in)
            action = self.pi.act(mem_emb)

        return action.item(), hid_out

    def explore(self, obs, prev_act, prev_rew, hid_in):
        with torch.no_grad():
            mem_emb, hid_out = self.memory(obs, prev_act, prev_rew, hid_in)
            action, _, _ = self.pi.sample(mem_emb)

        return action.item(), hid_out
