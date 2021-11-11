import torch
import torch.nn as nn

from metanas.meta_optimizer.agents.core import mlp


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size=256, layers=2,
                 activation=nn.ReLU):
        super().__init__()
        self.a = mlp([obs_dim] + [hidden_size]*layers + [act_dim], activation)
        self.v = mlp([obs_dim] + [hidden_size]*layers + [1], activation)

    def forward(self, obs):
        a = self.a(obs)
        v = self.v(obs)
        return v + a - a.mean()


class LSTMQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU):
        super().__init__()
        self.activation = activation()

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.a = nn.Linear(hidden_size, act_dim)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, h, c):
        x = self.activation(self.linear1(x))
        x, (h_out, c_out) = self.lstm(x, (h, c))

        v = self.v(x)
        a = self.a(x)

        return v + a - a.mean(), h_out, c_out


class RL2QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, device, hidden_size=64,
                 activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.act_dim = act_dim
        self.device = device

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size+act_dim+1,  # +1 for the reward
                          hidden_size,
                          batch_first=True)

        self.a = nn.Linear(hidden_size, act_dim)
        self.v = nn.Linear(hidden_size, 1)

    def _one_hot(self, act):
        return torch.eye(self.act_dim)[act.long(), :].to(self.device)

    def forward(self, obs, prev_act, prev_rew, hid_in,
                training=False):

        prev_act = self._one_hot(prev_act)
        obs_enc = self.activation(self.linear1(obs))

        gru_input = torch.cat(
            [
                obs_enc,
                prev_act,
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

        v = self.v(gru_out)
        a = self.a(gru_out)
        return v + a - a.mean(), hid_out
