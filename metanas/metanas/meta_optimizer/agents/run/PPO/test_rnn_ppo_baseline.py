import argparse

from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs
from metanas.meta_optimizer.agents.utils.env_wrappers import CartPolePOMDPWrapper

import gym

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


if __name__ == "__main__":
    # Garage implements RL2 instead of LSTM-PPO
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_pomdp", action="store_true",
                        help="Use POMDP CartPole environment")
    args = parser.parse_args()

    # multiprocess environment
    # env = make_vec_env('CartPole-v1', n_envs=4)

    path = "CartPole/RL2_PPO"
    if args.use_pomdp:
        path = "CartPole/RL2_PPO_POMDP"
        env = CartPolePOMDPWrapper(gym.make("CartPole-v1"))
    else:
        env = gym.make("CartPole-v1")

    logger_kwargs = setup_logger_kwargs(path, seed=args.seed)

# logger_kwargs['output_dir']
    model = PPO2(MlpLstmPolicy, env, verbose=1,
                 tensorboard_log=logger_kwargs['output_dir'])
    model.learn(total_timesteps=100*4000)
