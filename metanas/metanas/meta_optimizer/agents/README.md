# Reinforcement Learning Agents
The recurrent reinforcement learning agents to pre-optimize the architecture search space of DARTS for metaNAS.

# Implemented Agent
All agents employ RL^2 policy and value function networks, which use recurrent networks and take the current observation, last action and last reward as input. All agents are applied to discrete action space and illegal action masking.

- Random Policy 
- Proximal Policy Optimization

## References
- https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
- https://github.com/openai/spinningup