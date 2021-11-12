import argparse

from metanas.meta_optimizer.agents.SAC.sac import SAC
from metanas.meta_optimizer.agents.PPO.ppo import PPO
from metanas.meta_optimizer.agents.DQN.dqn import DQN
from metanas.meta_optimizer.agents.DQN.drqn import DRQN
from metanas.meta_optimizer.agents.PPO.rl2_ppo import PPO as RL2_PPO
from metanas.meta_optimizer.agents.SAC.rl2_sac import SAC as RL2_SAC
from metanas.meta_optimizer.agents.DQN.rl2_dqn import DQN as RL2_DQN
from metanas.env.env_wrappers import CartPolePOMDPWrapper
from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs

import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_pomdp", action="store_true",
                        help="Use POMDP CartPole environment")
    parser.add_argument(
        "--agent", type=str, default="PPO",
        help="RL2_PPO/SAC/RL2_SAC/DQN/DRQN/RL2_DQN")
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

    # Some algorithms might not run due to changes of parameters
    if args.agent == "PPO":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = PPO(None, env,
                    epochs=args.epochs,
                    hidden_size=args.hidden_size,
                    logger_kwargs=logger_kwargs)
    elif args.agent == "RL2_PPO":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = RL2_PPO(None, env,
                        epochs=args.epochs,
                        hidden_size=args.hidden_size,
                        logger_kwargs=logger_kwargs)
    elif args.agent == "SAC":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = SAC(None, env, test_env, ac_kwargs=ac_kwargs,
                    epochs=args.epochs,
                    logger_kwargs=logger_kwargs)
    elif args.agent == "RL2_SAC":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = RL2_SAC(None, env, ac_kwargs=ac_kwargs,
                        epochs=args.epochs,
                        hidden_size=args.hidden_size,
                        logger_kwargs=logger_kwargs)
    elif args.agent == "DQN":
        qnet_kwargs = dict(hidden_size=args.hidden_size)
        agent = DQN(None, env, test_env,
                    qnet_kwargs=qnet_kwargs,
                    epochs=args.epochs,
                    logger_kwargs=logger_kwargs)
    elif args.agent == "DRQN":
        qnet_kwargs = dict(hidden_size=args.hidden_size)
        agent = DRQN(None, env, test_env,
                     qnet_kwargs=qnet_kwargs,
                     epochs=args.epochs,
                     logger_kwargs=logger_kwargs)
    elif args.agent == "RL2_DQN":
        qnet_kwargs = dict(hidden_size=args.hidden_size)
        agent = RL2_DQN(None, env, qnet_kwargs=qnet_kwargs,
                        epochs=args.epochs,
                        logger_kwargs=logger_kwargs)
    else:
        raise ValueError(f"The given agent {args.agent} is not supported")

    agent.train_agent(env)
