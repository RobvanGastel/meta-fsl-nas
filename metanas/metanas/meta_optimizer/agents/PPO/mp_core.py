import torch
import torch.nn as nn

import numpy as np

from torch.distributions.categorical import Categorical
from metanas.meta_optimizer.agents.PPO.core import (layer_init,
                                                    MaskedCategorical)


class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size, device, sequence_length,
                 activation=nn.ReLU, use_mask=True):
        super().__init__()

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.hidden_size = hidden_size
        self.activation = activation()
        self.sequence_length = sequence_length
        self.use_mask = use_mask
        self.device = device
        self.act_dim = act_dim

        # Encoding the observations + orthogonal initialization
        self.obs_enc = layer_init(nn.Linear(obs_dim, self.hidden_size))

        # Gru input: (batch size, obs embedding + one-hot actions + reward)
        self.gru = nn.GRU(self.hidden_size+act_dim+1,
                          self.hidden_size,
                          batch_first=True)

        # Orthogonal initialization of the RNN layer
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        # Policy
        self.linear_pi = layer_init(
            nn.Linear(self.hidden_size, act_dim), std=np.sqrt(0.01))

        # Value function
        self.linear_v = layer_init(
            nn.Linear(self.hidden_size, 1), std=1)

    def _one_hot(self, act):
        return torch.eye(self.act_dim)[act.long(), :].to(self.device)

    def pi(self, obs, prev_act, prev_rew, hid_in, mask=None, action=None,
           training=False):
        """Obtain the π(obs, prev_act, prev_rew) for the given sequences.

        Args:
            obs (torch.tensor): Sequence of observations
            prev_act (torch.tensor): Sequence of previous actions
            prev_rew (torch.tensor): Sequence of previous rewards
            hid_in (torch.tensor): RNN hidden states
            mask (torch.tensor, optional): Action mask. Defaults to None.
            action (torch.tensor, optional): Given action for Pi. Defaults
            to None.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            torch.tensor: Policy π
            torch.tensor: Log probability π(a)
            torch.tensor: RNN hidden states output
        """
        # previous action one-hot encoding: (batch_size, act_dim)
        prev_act = self._one_hot(prev_act)
        obs_enc = self.activation(self.obs_enc(obs))

        gru_input = torch.cat(
            [
                obs_enc,
                prev_act,
                prev_rew
            ],
            dim=-1
        )

        if training:
            # Input rnn: (batch size, sequence length, features)
            h = gru_input.size()
            gru_input = gru_input.reshape(
                (h[0]//self.sequence_length), self.sequence_length, h[1])

            gru_out, hid_out = self.gru(gru_input, hid_in)

            h = gru_out.size()
            gru_out = gru_out.reshape(h[0] * h[1], h[2])
            # Output rnn: (batch size, features)
        else:
            # Input rnn: (1, 1, features)
            gru_input = gru_input.unsqueeze(1)
            gru_out, hid_out = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(1)
            # Output rnn: (1, features)

        logits = self.linear_pi(gru_out)

        # Action masking
        if self.use_mask:
            pi = MaskedCategorical(logits=logits, mask=mask)
        else:
            pi = Categorical(logits=logits)

        logp_a = None
        if action is not None:
            logp_a = pi.log_prob(action)
        return pi, logp_a, hid_out

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
            training (bool): Training mode. Defaults to False.

        Returns:
            torch.tensor: State-value function estimates
        """

        prev_act = self._one_hot(prev_act)
        obs_enc = self.activation(self.obs_enc(obs))

        gru_input = torch.cat(
            [
                obs_enc,
                prev_act,
                prev_rew
            ],
            dim=-1
        )

        if training:
            # Input rnn: (batch size, sequence length, features)
            h = gru_input.size()
            gru_input = gru_input.reshape(
                (h[0]//self.sequence_length), self.sequence_length, h[1])

            gru_out, hid_out = self.gru(gru_input, hid_in)

            h = gru_out.size()
            gru_out = gru_out.reshape(h[0] * h[1], h[2])
            # Output rnn: (batch size, features)
        else:
            # Input rnn: (1, 1, features)
            gru_input = gru_input.unsqueeze(1)
            gru_out, _ = self.gru(gru_input, hid_in)
            gru_out = gru_out.squeeze(1)
            # Output rnn: (1, features)

        v = self.linear_v(gru_out).reshape(-1)
        return v

    def step(self, obs, prev_act, prev_rew, hid_in, mask=None):
        """Obtain actor-critic step values for the environment
        without gradient computation.

        Args:
            obs (torch.tensor): sequence of observations
            prev_act (torch.tesnor): sequence of previous actions
            prev_rew (torch.tensor): sequence of previous rewards
            hid_in (torch.tensor): hidden weights of the RNN
            mask (torch.tensor, optional): action mask. Defaults to None.

        Returns:
            torch.tensor: Sampled action from π
            torch.tensor: State-value function estimates
            torch.tensor: Log probability of π(a)
            torch.tensor: RNN hidden states ouput
        """

        with torch.no_grad():
            pi, _, hid_out = self.pi(
                obs, prev_act, prev_rew, hid_in, mask)
            action = pi.sample()

            # Log_prob of action a
            logp_a = pi.log_prob(action)
            v = self.v(obs, prev_act, prev_rew, hid_in)

        return action.cpu(), v.cpu(), logp_a.cpu(), hid_out
