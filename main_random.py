import argparse
import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import numpy as np
from stable_baselines3.common.utils import set_random_seed
import torch
from stable_baselines3.td3.policies import TD3Policy
import matplotlib.pyplot as plt

# Setting up the argument parser for command-line inputs
parser = argparse.ArgumentParser()

# Device to run the code on (CPU or GPU)
parser.add_argument('--device', type=str, default='cpu', help='Specify device: cpu or cuda')

# Create path to save and load
# parser.add_argument('--path', type=str, default=None, help='Folder path without (/) in the end, specify load and save path each time')

# Whether to load a pre-trained model
parser.add_argument('--load', type=str, default='True', help='Load a pre-trained model (True/False)', choices=['True', 'False'])

# Path to save the trained model
parser.add_argument('--save', default=True, help='Save the model or not')

# Number of episodes for evaluation
parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')

# Total timesteps to train the model
parser.add_argument('--train_timesteps', type=int, default=None, help='Number of timesteps for training (set to None to skip training)')

# Maximum steps allowed per episode
parser.add_argument('--max_episode_step', type=int, default=None, help='Maximum steps allowed per episode.')

# Fix seed
parser.add_argument('--seed', type=int, default=3, help='Random seed to guarantee reproducibility')
#seed 3
# Parse the input arguments
args = parser.parse_args()

# Extract parsed arguments for convenience
device = args.device
# model_path = args.path
load = args.load
train_timesteps = args.train_timesteps
eval_episodes = args.eval_episodes
max_episode_step = args.max_episode_step
save = args.save
seed = args.seed

set_random_seed(seed)

# Register the custom CartPoleSwingUp environment with Gym

gym.register(
    id='CartPoleSwingUpRandom',
    entry_point='myCartpoleF_random:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)


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


# Function to create env instance with randomized paramters from sensitivity analysis
def mod_make_env(env_id, param_dict, render_mode=None):
    def _init():
        # from registered gym env call make
        # env = gym.make('CartPoleSwingUpRandom',  render_mode='human')
        env = gym.make(env_id, render_mode = render_mode)
        return RandomizedEnv(env, param_dict)
    return _init

#TODO: custom callback for tensorboad loggin!
    

# Create a vectorized environment with rendering enabled
num_envs = 10

envs = DummyVecEnv([mod_make_env('CartPoleSwingUpRandom', params) for _ in range(num_envs)])

# Log env parameters
# current_params = [env.env.env.current_params for env in envs.envs]
# for env in env_parameters:
# print("\nRandomized Parameters per Environment:")
# for i, p in enumerate(current_params):
#     
#     print(f"Env {i+1}:")
#     print(p.items())
#     for key, value in p.items():
#         print(f" {key:<10}: {value:.2E}")
#     print("-"*20)
# 

# Load or initialize the TD3 model

if load == 'True':
    print("Loading the pre-trained model...")
    model = TD3.load(path='model/td3_swingup_balance', env=envs)
    model.load_replay_buffer("model/td3_swingup_balance_replay_buffer")
else:
    # Add noise to actions for exploration during training
    action_noise = NormalActionNoise(
        mean=np.zeros(envs.action_space.shape),
        sigma=0.1 * np.ones(envs.action_space.shape)
    )

    # Initialize a new TD3 model with a custom neural network architecture

    # Create the TD3 model with the custom policy
    model = TD3(
        TD3Policy,
        envs,
        policy_kwargs=dict(net_arch=[512, 256, 24]),
        action_noise=action_noise, 
        verbose=0
    )

# custom callback to render during training
from stable_baselines3.common.callbacks import BaseCallback

class ExternalRenderCallback(BaseCallback):
    def __init__(self, render_env): # render_freq = 1000
        super().__init__()
        self.render_env = render_env
        # self.render_freq = render_freq
        self.obs = self.render_env.reset()

    def _on_step(self) -> bool:
        # if self.n_calls % self.render_freq == 0:
        action, _ = self.model.predict(self.obs, deterministic = True)
        self.obs, _, dones, _ = self.render_env.step(action)
        
        self.render_env.envs[0].render()

        if dones[0]:
            self.render_env.envs[0].reset()
        return True

# render_env = gym.make('CartPoleSwingUpRandom', render_mode = 'human')
# render_env = RandomizedEnv(render_env, params)
# render_env = envs
render_env = DummyVecEnv([mod_make_env('CartPoleSwingUpRandom', params, render_mode='human') for _ in range(int(num_envs / num_envs))])

# Train the model if the user specifies a training duration
if train_timesteps is not None:
    print('--------------Training the Model--------------')
    model.learn(total_timesteps=train_timesteps, 
                callback = ExternalRenderCallback(render_env))


# Save the trained model if a save path is specified
if save:
    print(f"Saving the model")
    model.save('model/td3_swingup_balance')
    model.save_replay_buffer('model/td3_swingup_balance_replay_buffer')

# Evaluate the model
if eval_episodes is not None:
    print('--------------Evaluating the Model--------------')
    # env = gym.make('CartPoleSwingUpRandom', render_mode='human')
    env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom', render_mode='human')])
    for episode in range(eval_episodes):
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
            i += 1
            # Predict the action
            action, _ = model.predict(obs, deterministic=False)
            force_list.append(10 * action[0])
            # Take the action and observe the result
            obs, reward, done, info = env.step(action)
            cos_list.append(obs[0][2])
            x_dot_list.append(obs[0][1])
            theta_dot_list.append(obs[0][4])
            # Add the reward to the total
            total_reward += reward
            # Render the environment
            env.render()
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
        plt.show()
# Close the environment
env.close()
