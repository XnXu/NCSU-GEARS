from __future__ import annotations
import random
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn

# from scipy.integrate import solve_ivp
# from scipy.integrate import odeint
# import scipy

import gymnasium as gym
import argparse
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.td3.policies import TD3Policy
import matplotlib
matplotlib.use('Agg') # non interactive backend to save plots
import matplotlib.pyplot as plt

## this test adds noise to obs and action
## it also adds noise to system paramters

# Setting up the argument parser for command-line inputs
parser = argparse.ArgumentParser()

# Whether to load a pre-trained model
parser.add_argument('--load', type=str, default='True', help='Load a pre-trained model (True/False)', choices=['True', 'False'])

# Number of episodes for evaluation
parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')

# Maximum steps allowed per episode
parser.add_argument('--max_episode_step', type=int, default=3000, help='Maximum steps allowed per episode.')

# Standard deviation of additive Gaussian noise
parser.add_argument('--noise_std', type=float, default=0.05, help='Additive noise standard deviation to obs and action')

parser.add_argument('--seed', type=int, default=3, help='Random seed to guarantee reproducibility')
#seed 3
# Parse the input arguments
args = parser.parse_args()

# Extract parsed arguments for convenience

load = args.load
eval_episodes = args.eval_episodes
max_episode_step = args.max_episode_step
seed = args.seed
noise_std = args.noise_std
set_random_seed(seed)

# Parameters of environment as dict
# keys name of the parameters in env
# items range of parameters to be uniformly chosen from
params = {
        "Jm": (3.783e-7, 4.017e-7),
        "Beq": (5.238, 5.562),
        "Bp": (2.328e-3, 2.472e-3),
        }

# Custom Wrapper for Randomization
class RandomizedEnv(gym.Wrapper):
    def __init__(self, env, param_ranges):
        """ add intervals on each uncertain parameters"""
        super().__init__(env)
        self.param_ranges = param_ranges
        self.current_params = {}

    def set_params(self):
        """ randomize and apply new parameters on each reset"""
        self.current_params = {
            key: np.random.uniform(low, high) for key, 
            (low, high) in self.param_ranges.items()
        }
        print(f"RandPars_set: {self.current_params}")
        for key, value in self.current_params.items():
            if hasattr(self.env.unwrapped, key):
                setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs):
        self.set_params()
        print(f"RandPars_after_reset: {self.current_params}")
        return self.env.reset(**kwargs)

# Function to create env instance with randomized paramters from sensitivity analysis
def mod_make_env(env_id, param_dict):
    def _init():
        # from registered gym env call make
        # env = gym.make('CartPoleSwingUpRandom',  render_mode='human')
        env = gym.make(env_id, render_mode = 'human')
        return RandomizedEnv(env, param_dict)
    return _init

gym.register(
    id='CartPoleSwingUpRandom',
    entry_point='myCartpoleF_random:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)

env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom',render_mode='human')])
# print(gym.spec('CartPoleSwingUpRandom'))

## Add noise to observation

def add_noise_obs(obs, noise_std = noise_std):
    return obs + np.random.normal(0, noise_std, size = obs.shape)

## Add noise to action
def add_noise_action(action, noise_std = noise_std):
    return np.clip(action + np.random.normal(0, noise_std, size = action.shape), -1, 1) # clip to normalized action (-1,1) same as env


episodes = 10

if load == 'True':
    print("Loading the pre-trained model...")
    model = TD3.load(path='model/td3_swingup_balance', env=env)
    model.load_replay_buffer("model/td3_swingup_balance_replay_buffer")
else:
    # Add noise to actions for exploration during training
    action_noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=0.1 * np.ones(env.action_space.shape)
    )


# Evaluate the model

if eval_episodes is not None:
    print('------Evaluating Control Without Model Mismatch------')
    # TODO: add mod_make_env as a function here to randomize env params
    # env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom',render_mode='human')])
    for episode in range(eval_episodes):
        env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom',render_mode='human')])
        # Reset the environment at the start of each episode
        obs = env.reset()
        done = False
        # Accumulate rewards for this episode
        total_reward = 0
        force_list = []
        cos_list = []
        x_dot_list = []
        theta_dot_list = []
        i = 0
        while not done and i < 1000:
            noisy_obs = add_noise_obs(obs)
            i += 1
            # Predict the action
            action, _ = model.predict(noisy_obs)
            noisy_action = add_noise_action(action[0])
            force_list.append(10 * noisy_action[0])
            # Take the action and observe the result
            obs, reward, done, info = env.step(noisy_action)
            cos_list.append(obs[0][2])
            x_dot_list.append(obs[0][1])
            theta_dot_list.append(obs[0][4])
            # Add the reward to the total
            total_reward += reward
            # Render the environment
            env.render()
        
        # env.close()
        print(f'Episode: {episode + 1} | Total Reward: {total_reward}')
        fig, axes = plt.subplots(4, 1, figsize=(8, 12))  # 4 rows, 1 column

        axes[0].plot(cos_list)
        axes[0].set_ylabel('Cosine theta')

        axes[1].plot(x_dot_list)
        axes[1].set_ylabel('x_dot')

        axes[2].plot(theta_dot_list)
        axes[2].set_ylabel('theta_dot')

        axes[3].plot(force_list)
        axes[3].set_ylabel('force')

        plt.tight_layout()  # Adjusts layout to prevent overlapping labels
        plt.savefig("model/test_trajectories_250226.png", dpi=300)
       
env.close()
