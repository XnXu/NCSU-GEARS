from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn

# from scipy.integrate import solve_ivp
# from scipy.integrate import odeint

# import scipy

import sys
import gymnasium as gym
import time
import signal

balance_time = 240; h_in = 1/100;
CartPoleSwingUp = gym.register(id='CartPoleSwingUp',
  #                           entry_point='gymnasium.envs.classic_control.myCartpoleF_SwingUp:CartPoleSwingUp',
                             entry_point='myCartpoleF_SwingUp:CartPoleSwingUp',
                             reward_threshold=-40*0.95,
                             max_episode_steps=int( balance_time / h_in ),
                               )
env = gym.make('CartPoleSwingUp',render_mode='human')
print(gym.spec('CartPoleSwingUp'))

episodes = 10

for episode in range(episodes):
    observation = env.reset()[0]  # Initialize the environment

    done = False
    while not done:
        env.render()  # Visualize the environment
        x, x_dot, cos, sin, theta_dot = observation
        print(x, x_dot, cos, sin, theta_dot)
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, done, _ = env.step(action)
        #print(f'{observation, reward, terminated, done}')

env.close()
#DDPG PPD