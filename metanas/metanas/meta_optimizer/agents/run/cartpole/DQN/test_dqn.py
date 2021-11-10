import argparse

from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs
from metanas.meta_optimizer.agents.DQN.dqn import DQN
from metanas.meta_optimizer.agents.utils.env_wrappers import CartPolePOMDPWrapper

import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_pomdp", action="store_true",
                        help="Use POMDP CartPole environment")
    args = parser.parse_args()

    path = "CartPole/DQN"
    if args.use_pomdp:
        path = "CartPole/DQN_POMDP"
        env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
        test_env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
    else:
        env = gym.make("CartPole-v1")
        test_env = gym.make("CartPole-v1")

    env.max_ep_len = 500
    test_env.max_ep_len = 500

    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    qnet_kwargs = dict(hidden_size=128, layers=2)
    agent = DQN(None, env, test_env, seed=args.seed, qnet_kwargs=qnet_kwargs,
                logger_kwargs=logger_kwargs)
    agent.train_agent()
