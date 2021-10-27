import argparse

from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs
from metanas.meta_optimizer.agents.SAC.sac import SAC
from metanas.meta_optimizer.agents.utils.env_wrappers import CartPolePOMDPWrapper

import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_pomdp", action="store_true",
                        help="Use POMDP CartPole environment")
    args = parser.parse_args()

    path = "CartPole/SAC"
    if args.use_pomdp:
        path = "CartPole/SAC_POMDP"
        env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
        test_env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
    else:
        env = gym.make("CartPole-v1")
        test_env = gym.make("CartPole-v1")

    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    ac_kwargs = dict(hidden_size=[128]*2)
    agent = SAC(env, test_env, ac_kwargs=ac_kwargs,
                logger_kwargs=logger_kwargs)
    agent.train_agent()
