import random
import argparse

import numpy as np

from metanas.env.krazy_world import KrazyGridWorld
from metanas.meta_optimizer.agents.PPO.rl2_ppo import PPO
from metanas.meta_optimizer.agents.SAC.rl2_sac import SAC
from metanas.meta_optimizer.agents.DQN.rl2_dqn import DQN
from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs


def init_krazyworld(task_seed):

    # The settings except the seeds are taken from the E-MAML
    # implementation of krazyworld.
    return KrazyGridWorld(
        screen_height=256, grid_squares_per_row=10,
        one_hot_obs=False, use_local_obs=True, image_obs=False,
        seed=42, task_seed=task_seed,
        num_goals=3, max_goal_distance=np.inf, min_goal_distance=2,
        death_square_percentage=0.08,
        num_steps_before_energy_needed=50, energy_sq_perc=0.05,
        energy_replenish=8, num_transporters=1,
        ice_sq_perc=0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="PPO", help="PPO/SAC/DQN")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--iterations", type=int,
                        default=2000, help="meta-training iterations")
    args = parser.parse_args()

    # Distribution, same number of testing and training envs as used
    # in E-MAML
    envs = [init_krazyworld(seed) for seed in range(32)]
    test_envs = [init_krazyworld(seed) for seed in range(100, 164)]

    # Setup logging
    path = f"KrazyWorld/RL2_${args.agent}"
    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

    env = random.sample(envs, k=1)[0]
    if args.agent == "PPO":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = PPO(None, env, hidden_size=args.hidden_size,
                    logger_kwargs=logger_kwargs)
    if args.agent == "SAC":
        ac_kwargs = dict(hidden_size=[args.hidden_size]*2)
        agent = SAC(None, env, ac_kwargs=ac_kwargs,
                    lr=3e-4,
                    epochs=1,
                    hidden_size=args.hidden_size,
                    steps_per_epoch=4000,
                    start_steps=10000,
                    update_after=1000,
                    update_every=20,
                    batch_size=32,
                    replay_size=int(1e6),
                    seed=42,
                    num_test_episodes=10,
                    logger_kwargs=logger_kwargs)
    if args.agent == "DQN":
        qnet_kwargs = dict(hidden_size=args.hidden_size)

        env = random.sample(envs, k=1)[0]
        agent = DQN(None, env, qnet_kwargs=qnet_kwargs,
                    epochs=100, seed=42,
                    logger_kwargs=logger_kwargs)

    agent.train_agent()

    # Meta-loop
    # for i in range(args.iterations):
    #     env = random.sample(envs, k=1)[0]
    #     agent.train_agent(env)

    # For loop close the environments
    # env.close()
