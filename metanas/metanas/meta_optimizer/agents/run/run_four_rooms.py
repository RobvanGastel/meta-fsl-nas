import argparse

import numpy as np

from metanas.env.four_room_env import FourRoomsEnv
from metanas.meta_optimizer.agents.PPO.rl2_ppo import PPO
from metanas.meta_optimizer.agents.SAC.rl2_sac import SAC
from metanas.meta_optimizer.agents.DQN.rl2_dqn import DQN
from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="PPO", help="PPO/SAC/DQN")
    parser.add_argument("--name", type=str, default="", help="experiment name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_meta_learning", action="store_true")
    parser.add_argument("--iterations", type=int,
                        default=2000, help="meta-training iterations")
    parser.add_argument("--test_trial", type=int,
                        default=20, help="execute test trial every n")

    parser.add_argument("--reset_buffer", action="store_true")
    parser.add_argument("--exploration_sampling", action="store_true")

    args = parser.parse_args()

    # Very simple meta-learning tasks
    envs = [FourRoomsEnv(seed=s) for s in range(32)]
    test_envs = [FourRoomsEnv(seed=s) for s in range(100, 164)]

    # Setup logging
    path = f"FourRooms/RL2_{args.agent}_{args.name}"
    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    env = np.random.choice(envs, 1)[0]
    if args.agent == "PPO":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = PPO(None, env,
                    epochs=args.epochs,
                    hidden_size=args.hidden_size,
                    exploration_sampling=args.exploration_sampling,
                    logger_kwargs=logger_kwargs)
    if args.agent == "SAC":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = SAC(None, env, ac_kwargs=ac_kwargs,
                    epochs=args.epochs,
                    reset_buffer=args.reset_buffer,
                    hidden_size=args.hidden_size,
                    exploration_sampling=args.exploration_sampling,
                    logger_kwargs=logger_kwargs)
    if args.agent == "DQN":
        qnet_kwargs = dict(hidden_size=args.hidden_size)
        agent = DQN(None, env, qnet_kwargs=qnet_kwargs,
                    epochs=args.epochs,
                    logger_kwargs=logger_kwargs)

    if args.use_meta_learning:
        for i in range(args.iterations):
            env = np.random.choice(envs, 1)[0]
            agent.train_agent(env)

            # if i % args.test_trial == 0:
            #     test_env = np.random.choice(test_envs, 1)[0]
            #     agent.test_agent(test_env)
    else:
        agent.train_agent(env)

    # For loop close the environments
    for env in envs+test_envs:
        env.close()
