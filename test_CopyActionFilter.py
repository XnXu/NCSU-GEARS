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
parser.add_argument('--load_model', type=str, default=None, help='Load a pre-trained model')

# Save to tensorboard log dir
parser.add_argument('--save_test', type=str, default=None, help='Save plots to tensorboard log dir')

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

load_model = args.load_model
save_test = args.save_test
eval_episodes = args.eval_episodes
max_episode_step = args.max_episode_step
seed = args.seed
noise_std = args.noise_std
set_random_seed(seed)
        
# Parameters of environment as dict
# keys name of the parameters in env
# items range of parameters to be uniformly chosen from
params = {
        "Jm": (3.78300e-7, 4.01700e-7),
        "Beq": (5.23800, 5.56200), 
        "Bp": (2.32800e-3, 2.47200e-3),
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
                key: np.random.uniform(low, high) for key, (low, high) in self.param_ranges.items() 
        }
        # print(f"RandPars_set: {self.current_params}")
        for key, value in self.current_params.items():
            if hasattr(self.env.unwrapped, key):
                # print(f"Parameter: {key}, CurrentValue: {value}")
                setattr(self.env.unwrapped, key, value)
                # print(f"Parameter: {key}, AfterResetValue: {value}")

    def reset(self, **kwargs):
        self.set_params()
        # print(f"RandPars_after_reset: {getattr(self.env.unwrapped, 'Jm')}")
        return self.env.reset(**kwargs)


# Function to create env instance with randomized parameters from sensitivity analysis
def mod_make_env(env_id, param_dict, render_mode=None):
    def _init():
        # from registered gym env call make
        # env = gym.make('CartPoleSwingUpRandom',  render_mode='human')
        env = gym.make(env_id, render_mode = render_mode)
        # rand_env = RandomizedEnv(env, param_dict)
        return RandomizedEnv(env, param_dict)
        # return ActionFilterWrapper(rand_env, filter_tau, tau)
    return _init
    
gym.register(
    id='CartPoleSwingUpRandom',
    entry_point='myCartpoleF_random:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)

# randomize env system parameters
num_envs = 1

env = DummyVecEnv([mod_make_env('CartPoleSwingUpRandom', params, render_mode='human') for _ in range(num_envs)])
# env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom', render_mode='human')])

# get env attributes: tau, filter_tau, params
filter_tau=np.float32(env.get_attr("filter_tau"))
tau=np.float32(env.get_attr("tau"))

def add_noise_obs(obs, noise_std = noise_std):
    return obs + np.random.normal(0, noise_std, size = obs.shape)

## Add noise to action
def add_noise_action(action, noise_std = noise_std):
    return np.clip(action + np.random.normal(0, noise_std, size = action.shape), -1, 1) # clip to normalized action (-1,1) same as env

if load_model is not None:
    print("Loading the pre-trained model...")
    model = TD3.load(path=load_model, env=env)
    model.load_replay_buffer(load_model + "_replay_buffer")
else:
    # Add noise to actions for exploration during training
    print("Add model to load for testing")

def run_episode(env, filter_tau, tau=0.02, max_episode_step=1000):
    total_reward = 0
    obs = env.reset()
    # check env_params after reset
    env_params={key: np.float64(env.get_attr(key)[0]) for key, value in params.items()}
    
    ep_step_counter = 0
    raw_force = np.zeros((max_episode_step, 1))
    recorded_raw_force = np.zeros((max_episode_step, 1))
    filtered_force = np.zeros((max_episode_step, 1))
    obs_history = np.zeros((max_episode_step, obs[0].shape[0]))
    
    # observe without added noise
    for i in range(max_episode_step):
        # add noise to obs
        noisy_obs = add_noise_obs(obs)
        # Predict the action
        action, _ = model.predict(noisy_obs)
        raw_force[i] = (10 * action[0].copy())
        # add noise to action
        noisy_action = add_noise_action(action)
        # step based on noisy action
        obs, reward, done, info = env.step(noisy_action)
        total_reward += reward
        recorded_raw_force[i] = (info[0]['raw'])
        filtered_force[i] = (info[0]['filtered'])
        obs_history[i,:] = obs[0]
        ep_step_counter += 1
        
        if done: 
            break
    # env.close()    
    
    # convert history to np array for plotting
    # raw_force = np.array(env.get_attr("raw_control_history")[-2])
    # filtered_force = np.array(env.get_attr("filtered_control_history")[-2])
    # state_trajectories = np.array(env.get_attr("state_history")[-1]).squeeze()
    return raw_force, recorded_raw_force, filtered_force, obs_history, total_reward, ep_step_counter, env_params
    
for episode in range(eval_episodes):
    raw, recorded_raw, filtered, obs_history, total_reward, ep_step_counter, env_params = run_episode(env, filter_tau, tau, max_episode_step)
    num_steps = min(max_episode_step, ep_step_counter)
    print(f'Displaying Episode Length: {num_steps * tau} sec')
    tspan = np.linspace(0, tau * (num_steps), num_steps)
    print(f'Episode: {episode + 1} | Total Reward: {total_reward}')
    # print(raw[5])
    # print(filtered[5])
    # print(states)
    for key, value in env_params.items():
        print(f"{key}: Type= {type(value)}, Value= {value: .20f}")
    # plots
    fig, axes = plt.subplots(2, 3, figsize=(12,8))  # 2 rows, 3 column
    axes[0,0].plot(tspan, obs_history[0:num_steps, 2])
    axes[0,0].set_ylabel('Cosine theta')

    axes[0,1].plot(tspan, obs_history[0:num_steps,1])
    axes[0,1].set_ylabel('x_dot')

    axes[1,0].plot(tspan, obs_history[0:num_steps, -1])
    axes[1,0].set_ylabel('theta_dot')

    axes[1,1].plot(tspan, raw[0:num_steps])
    axes[1,1].set_ylabel('raw force')
    
    axes[0,2].plot(tspan, filtered[0:num_steps])
    axes[0,2].set_ylabel('filtered force')

    axes[1,2].plot(tspan, recorded_raw[0:num_steps])
    axes[1,2].set_ylabel('recorded raw force')
    
    plt.tight_layout()  # Adjusts layout to prevent overlapping labels
    
    plt.savefig(save_test + "test_" + str(episode) + "_noise=" + str(noise_std) + "_with_model_mismatch.png", dpi=300)

env.close()
