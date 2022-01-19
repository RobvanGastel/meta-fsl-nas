import torch
import torch.nn as nn

import numpy as np
import scipy.signal

from metanas.meta_optimizer.agents.core import mlp
from torch.distributions.categorical import Categorical


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)],
                                x[::-1], axis=0)[::-1]


def mean_no_none(l):
    l_no_none = [el for el in l if el is not None]
    return sum(l_no_none) / len(l_no_none)


def aggregate_dicts(dicts, operation=mean_no_none):
    all_keys = set().union(*[el.keys() for el in dicts])
    return {k: operation([dic.get(k, None) for dic in dicts]) for k in all_keys}


def aggregate_info_dicts(dicts):
    agg_dict = aggregate_dicts(dicts)

    return {k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in agg_dict.items()}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# source, https://boring-guy.sh/posts/masking-rl/
class MaskedCategorical(Categorical):

    def __init__(self, logits, mask=None):
        self.mask = mask
        if mask is None:
            super(MaskedCategorical, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(MaskedCategorical, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p,
                              torch.tensor(0.).to(p_log_p.device))
        return -p_log_p.sum(-1)

    def log_prob(self, action):
        if self.mask is None:
            return super().log_prob(action)

        log_prob = super().log_prob(action)
        return log_prob


"""Actor-Critic classes for RNN & MLPs"""


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPActorCritic(nn.Module):
    def __init__(self, env, hidden_size, activation=nn.ReLU):
        super().__init__()

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        hidden_sizes = [hidden_size] * 2
        self.pi = MLPCategoricalActor(
            obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size, device, activation=nn.ReLU):
        super().__init__()

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.hidden_size = hidden_size
        self.activation = activation()
        self.device = device
        self.act_dim = act_dim

        self.linear1 = layer_init(nn.Linear(obs_dim, self.hidden_size))

        # +1 for the reward
        self.gru = nn.GRU(self.hidden_size+act_dim+1,
                          self.hidden_size,
                          batch_first=True)

        self.linear_pi = layer_init(
            nn.Linear(self.hidden_size, act_dim), std=0.01)
        self.linear_v = layer_init(nn.Linear(self.hidden_size, 1), std=1)

    def _one_hot(self, act):
        return torch.eye(self.act_dim)[act.long(), :].to(self.device)

    def pi(self, obs, prev_act, prev_rew, hid_in, mask, action=None,
           training=False):

        # previous action one-hot encoding: (batch_size, act_dim)
        prev_act = self._one_hot(prev_act)
        obs_enc = self.activation(self.linear1(obs))

        gru_input = torch.cat(
            [
                obs_enc,
                prev_act,
                prev_rew
            ],
            dim=-1
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

        logits = self.linear_pi(gru_out)

        # Action masking
        pi = MaskedCategorical(logits=logits, mask=mask)

        logp_a = None
        if action is not None:
            logp_a = pi.log_prob(action)
        return pi, hid_out, logp_a

    def v(self, obs, prev_act, prev_rew, hid_in,
          training=False):
        """Value function approximation

        Args:
            obs (torch.tensor): Observations
            prev_act (torch.tensor): Previous actions
            prev_rew (torch.tensor): Previous reward
            hid_in (torch.tensor): RNN hidden states
            sequence_length (int, optional): Sequence length of
            the input. Defaults to 1.

        Returns:
            [torch.tensor]: Value-function estimates
        """

        prev_act = self._one_hot(prev_act)
        obs_enc = self.activation(self.linear1(obs))

        gru_input = torch.cat(
            [
                obs_enc,
                prev_act,
                prev_rew
            ],
            dim=-1
        )

        # Input rnn: (batch size, sequence length, features)
        if training:
            gru_input = gru_input.unsqueeze(0)
            gru_out, _ = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(0)
        else:
            gru_input = gru_input.unsqueeze(1)
            gru_out, _ = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(1)

        v = self.linear_v(gru_out).reshape(-1)
        return v

    def step(self, obs, prev_act, prev_rew, hid_in, mask):
        with torch.no_grad():
            # distribution

            pi, hid_out, _ = self.pi(obs, prev_act, prev_rew, hid_in, mask)
            a = pi.sample()

            # Log_prob of action a
            logp_a = pi.log_prob(a)
            v = self.v(obs, prev_act, prev_rew, hid_in)

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), hid_out

    def act(self, obs, prev_act, prev_rew, hid_in, mask):
        a, _, _, hid_out = self.step(obs, prev_act, prev_rew, hid_in, mask)
        return a, hid_out
