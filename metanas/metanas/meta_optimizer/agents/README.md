# Reinforcement Learning Agents
The recurrent reinforcement learning agents to find a better initialization for the alpha values of DARTS in metaNAS.

# Implemented Agents (WIP)
All agents employ RL^2 policy and value function networks, which use recurrent networks and take the current observation, last action and last reward as input. It should be noted that the implementation is for discrete action spaces.

Current implemented agents,
- Discrete-Soft Actor-Critic
- Deep Q-Networks
- Proximal Policy Optimization

## References
- [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763)
- [Discrete Soft-Actor Critic (SAC)](https://arxiv.org/abs/1910.07207)
- [Duelling DDQN](https://arxiv.org/abs/1511.06581)
- [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)