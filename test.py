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
eval_episodes = args.eval_episodes
max_episode_step = args.max_episode_step
seed = args.seed
noise_std = args.noise_std
set_random_seed(seed)


# Add low pass filter to action
class ActionFilterWrapper(gym.Wrapper):
    def __init__(self, env, filter_tau = 0.1, tau=0.01):
        """
        env: gym Env to wrap
        filter_tau: time constant for low-pass filter, filter_tau * v_out' + v_out = v_raw
        tau: time step same as env sample interval
        """
        super(ActionFilterWrapper, self).__init__(env)
        self.filter_tau = filter_tau
        self.tau = tau
        self.filtered_action = np.zeros(self.env.action_space.shape) if tau is not None else None
        # record signals and control performance
        self.state_history = []
        self.raw_action_history = []
        self.filtered_action_history = []
        
    def reset(self, **kwargs):
        obs = self.env.reset()
        if self.filtered_action is not None:
            self.filtered_action = np.zeros(self.env.action_space.shape)
        self.state_history = [np.array(obs[0]).copy()]
        self.raw_action_history = []
        self.filtered_action_history = []
        return obs
    
    def step(self, action):
        self.raw_action_history.append(action.copy()) # record raw action
        if self.filtered_action is not None:
            # apply low pass filter
            self.filtered_action += (self.tau / self.filter_tau) * (action - self.filtered_action)
            self.filtered_action_history.append(self.filtered_action.copy())
            # apply filtered action to step
            action_to_take = self.filtered_action
        else:
            # no filter applied
            action_to_take = action
        # apply action to env
        obs, reward, terminated, off_track, info = self.env.step(action_to_take)
        self.state_history.append(obs.copy())
        return obs, reward, terminated, off_track, info
        

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
                key: np.random.uniform(low, high) for key, (low, high) in self.param_ranges.items() 
        }
        # print(f"RandPars_set: {self.current_params}")
        for key, value in self.current_params.items():
            if hasattr(self.env.unwrapped, key):
                setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs):
        self.set_params()
        # print(f"RandPars_after_reset: {self.current_params}")
        return self.env.reset(**kwargs)


# Function to create env instance with randomized parameters from sensitivity analysis
def mod_make_env(env_id, param_dict, render_mode=None):
    def _init():
        # from registered gym env call make
        # env = gym.make('CartPoleSwingUpRandom',  render_mode='human')
        env = gym.make(env_id, render_mode = render_mode)
        # filtered_action_env = ActionFilterWrapper(env, filter_tau=0.1, tau=0.01)
        return RandomizedEnv(env, param_dict)
    return _init
    
# only test on original env
gym.register(
    id='CartPoleSwingUpRandom',
    entry_point='myCartpoleF_SwingUp:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)


# Create a vectorized environment with rendering enabled
num_envs = eval_episodes #TODO: should be num envs

envs = DummyVecEnv([mod_make_env('CartPoleSwingUpRandom', params) for _ in range(num_envs)])

# this works 
# env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom',render_mode='human')])
# print(gym.spec('CartPoleSwingUpRandom'))

## Add noise to observation

def add_noise_obs(obs, noise_std = noise_std):
    return obs + np.random.normal(0, noise_std, size = obs.shape)

## Add noise to action
def add_noise_action(action, noise_std = noise_std):
    return np.clip(action + np.random.normal(0, noise_std, size = action.shape), -1, 1) # clip to normalized action (-1,1) same as env


episodes = eval_episodes

if load_model is not None:
    print("Loading the pre-trained model...")
    model = TD3.load(path=load_model, env=envs)
    model.load_replay_buffer(load_model + "_replay_buffer")
else:
    # Add noise to actions for exploration during training
    print("Add model to load for testing")

# Evaluate the model
# Record eval

'''
from stable_baselines3.common.vec_env import VecVideoRecorder
vid_folder = "./model/videos/"
vid_length = max_episode_step
'''

if eval_episodes is not None:
    print('------Evaluating Control Without Model Mismatch------')
    # TODO: add mod_make_env as a function here to randomize env params
    # env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom',render_mode='human')])
    for episode in range(eval_episodes):
        env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom',render_mode='human')])
        # env = VecVideoRecorder(
        #         gym.make('CartPoleSwingUpRandom',render_mode='rbg_array'), 
        #                        vid_folder,
        #                        record_video_trigger=lambda t: t==0, 
        #                        video_length = vid_length)
        # Reset the environment at the start of each episode
        obs = env.reset()
        done = False
        # Accumulate rewards for this episode
        total_reward = 0
        force_list = []
        filtered_force_list = []
        cos_list = []
        x_dot_list = []
        theta_dot_list = []
        step_count = 0
        
        while not done and step_count < max_episode_step:
        # while not done:
        # if done.any():
            noisy_obs = add_noise_obs(obs)
            step_count += 1
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
            # if done.any():
            #     ep_count += sum(done)
            #     obs=env.reset()
        env.close()
        # print(f"Save episode {episode} to {vid_folder}")
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
        plt.savefig(load_model + str(episode) + "_noise=" + str(noise_std) + ".png", dpi=300)
       
env.close()


