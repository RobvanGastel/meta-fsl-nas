import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from metanas.meta_optimizer.agents.core import mlp


class CategoricalPolicy(nn.Module):
    def __init__(self, act_dim, hidden_size=64,
                 activation=nn.ReLU()):
        super().__init__()

        self.activation = activation

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, act_dim)

    def act(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))
        action_logits = self.policy(mem)

        # Greedy action selection
        return torch.argmax(action_logits, dim=-1)

    def sample(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))
        action_logits = self.policy(mem)

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

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.q = nn.Linear(hidden_size, act_dim)

    def forward(self, memory_emb):
        mem = self.activation(self.linear1(memory_emb))
        q = self.q(mem)
        return q


class Memory(nn.Module):
    def __init__(self, obs_dim, act_dim, device,
                 hidden_size=64, activation=nn.ReLU()):
        super().__init__()

        self.activation = activation
        self.act_dim = act_dim
        self.device = device

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        # +1 for the reward
        self.gru = nn.GRU(hidden_size+act_dim+1,
                          hidden_size,
                          batch_first=True)

    def _one_hot(self, act):
        return torch.eye(self.act_dim)[act.long(), :].to(self.device)

    def forward(self, obs, prev_act, prev_rew, hid_in,
                training=False):

        act_enc = self._one_hot(prev_act)
        obs_enc = self.activation(self.linear1(obs))

        gru_input = torch.cat(
            [
                obs_enc,
                act_enc,
                prev_rew
            ],
            dim=-1,
        )

        # Input rnn: (batch size, sequence length, features)
        if training:
            gru_input = gru_input.unsqueeze(0)
            gru_out, hid_out = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(0)
        else:
            gru_input = gru_input.unsqueeze(1)
            gru_out, hid_out = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(1)

        return gru_out, hid_out


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, device,
                 hidden_size=[256, 256], activation=nn.ReLU()):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n
        self.device = device

        self.memory = Memory(
            obs_dim, act_dim, device, hidden_size[0], activation=activation)
        self.pi = CategoricalPolicy(
            act_dim, hidden_size=hidden_size[0], activation=activation)

        self.q1 = QNetwork(obs_dim, act_dim,
                           hidden_size=hidden_size[0],
                           activation=activation)
        self.q2 = QNetwork(obs_dim, act_dim,
                           hidden_size=hidden_size[0],
                           activation=activation)

    def act(self, obs, prev_act, prev_rew, hid_in,
            training=False):
        with torch.no_grad():
            mem_emb, hid_out = self.memory(
                obs, prev_act, prev_rew, hid_in, training)
            action = self.pi.act(mem_emb)

        return action.item(), hid_out

    def explore(self, obs, prev_act, prev_rew, hid_in,
                training=False):
        with torch.no_grad():
            mem_emb, hid_out = self.memory(
                obs, prev_act, prev_rew, hid_in, training)
            action, _, _ = self.pi.sample(mem_emb)

        return action.item(), hid_out


class MLPCategoricalPolicy(nn.Module):
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

        self.pi = MLPCategoricalPolicy(
            obs_dim, act_dim, hidden_size, activation)
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
