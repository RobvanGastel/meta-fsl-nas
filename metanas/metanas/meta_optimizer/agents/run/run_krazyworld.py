import argparse

import numpy as np

from metanas.env.krazy_world_env import KrazyWorld
from metanas.meta_optimizer.agents.PPO.rl2_ppo import PPO
from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="PPO", help="PPO")
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

    # Distribution, same number of testing and training envs as used
    # in E-MAML
    envs = [KrazyWorld(seed=s, task_seed=s**2+1) for s in range(32)]
    test_envs = [KrazyWorld(seed=s, task_seed=s**2+1)
                 for s in range(100, 164)]

    # Setup logging
    path = f"KrazyWorld/RL2_{args.agent}_{args.name}"
    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    env = np.random.choice(envs, 1)[0]
    if args.agent == "PPO":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = PPO(None, env,
                    epochs=args.epochs,
                    hidden_size=args.hidden_size,
                    exploration_sampling=args.exploration_sampling,
                    logger_kwargs=logger_kwargs)

    if args.use_meta_learning:
        for i in range(args.iterations):
            env = np.random.choice(envs, 1)[0]
            agent.train_agent(env)

            if i % args.test_trial == 0:
                test_env = np.random.choice(test_envs, 1)[0]
                agent.test_agent(test_env)
    else:
        agent.train_agent(env)

    # For loop close the environments
    for env in envs+test_envs:
        env.close()
