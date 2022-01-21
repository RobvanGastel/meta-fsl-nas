import argparse
import numpy as np

from metanas.env.krazy_world_env import KrazyWorld
from metanas.meta_optimizer.agents.PPO.mp_rl2_ppo import PPO
from metanas.meta_optimizer.agents.Random.mp_random import RandomAgent
from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str,
                        default="PPO", help="PPO/Random")
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
    path = f"KrazyWorld/mp_{args.agent}_meta_learning_exploration_{args.name}"
    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    epochs = 3
    steps_per_worker = 1000
    env = np.random.choice(envs, 1)[0]

    if args.agent == "PPO":
        agent = PPO(None, [envs[0], envs[0]], epochs=epochs,
                    steps_per_worker=steps_per_worker,
                    exploration_sampling=True,
                    sequence_length=16, logger_kwargs=logger_kwargs)
    if args.agent == "Random":
        agent = RandomAgent(None, [envs[0], envs[0]],
                            steps_per_worker=steps_per_worker * epochs,
                            logger_kwargs=logger_kwargs)

    if args.use_meta_learning:
        for i in range(args.iterations):
            env = np.random.choice(envs, 1)[0]
            agent.set_task([env, env])
            agent.run_trial()

            if i % args.test_trial == 0:
                test_env = np.random.choice(test_envs, 1)[0]
                agent.set_task([test_env, test_env])
                agent.run_test_trial()
    else:
        agent.run_trial()

        # For loop close the environments
    for env in envs+test_envs:
        env.close()
