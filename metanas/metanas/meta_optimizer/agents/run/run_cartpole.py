import argparse

from metanas.meta_optimizer.agents.PPO.rl2_ppo import PPO as RL2_PPO
from metanas.env.env_wrappers import CartPolePOMDPWrapper
from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs

import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_pomdp", action="store_true",
                        help="Use POMDP CartPole environment")
    parser.add_argument(
        "--agent", type=str, default="RL2_PPO")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--epochs", type=int,
                        default=200, help="Epochs")
    args = parser.parse_args()

    # Setup logging
    path = f"KrazyWorld/${args.agent}"
    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    if args.use_pomdp:
        path += "_POMDP"
        env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
        test_env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
    else:
        env = gym.make("CartPole-v1")
        test_env = gym.make("CartPole-v1")

    env.max_ep_len = 500
    test_env.max_ep_len = 500

    if args.agent == "RL2_PPO":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = RL2_PPO(None, env,
                        epochs=args.epochs,
                        hidden_size=args.hidden_size,
                        logger_kwargs=logger_kwargs)
    else:
        raise ValueError(f"The given agent {args.agent} is not supported")

    agent.train_agent(env)
