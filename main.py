import argparse
import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3.common.utils import set_random_seed
import torch
from stable_baselines3.td3.policies import TD3Policy
import matplotlib.pyplot as plt

# Setting up the argument parser for command-line inputs
parser = argparse.ArgumentParser()

# Device to run the code on (CPU or GPU)
parser.add_argument('--device', type=str, default='cpu', help='Specify device: cpu or cuda')

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

parser.add_argument('--seed', type=int, default=3, help='Random seed to guarantee reproducibility')
#seed 3
# Parse the input arguments
args = parser.parse_args()

# Extract parsed arguments for convenience
device = args.device

load = args.load
train_timesteps = args.train_timesteps
eval_episodes = args.eval_episodes
max_episode_step = args.max_episode_step
save = args.save
seed = args.seed

set_random_seed(seed)

# Register the custom CartPoleSwingUp environment with Gym
gym.register(
    id='CartPoleSwingUp',
    entry_point='myCartpoleF_SwingUp:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)

# Create a vectorized environment with rendering enabled
env = DummyVecEnv([lambda: gym.make('CartPoleSwingUp', render_mode='human')])

# Load or initialize the TD3 model

if load == 'True':
    print("Loading the pre-trained model...")
    model = TD3.load(path='model/td3_smaller_force/td3_swingup_balance', env=env)
    model.load_replay_buffer("model/td3_smaller_force/td3_swingup_balance_replay_buffer")
else:
    # Add noise to actions for exploration during training
    action_noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=0.1 * np.ones(env.action_space.shape)
    )

    # Initialize a new TD3 model with a custom neural network architecture

    # Create the TD3 model with the custom policy
    model = TD3(
        TD3Policy,
        env,
        policy_kwargs=dict(net_arch=[256, 256, 32]),
        action_noise=action_noise
    )

# Train the model if the user specifies a training duration
if train_timesteps is not None:
    print('--------------Training the Model--------------')
    model.learn(total_timesteps=train_timesteps)

# Evaluate the model

if eval_episodes is not None:
    print('--------------Evaluating the Model--------------')
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
# Save the trained model if a save path is specified
if save:
    print(f"Saving the model")
    model.save('model/td3_nikki/td3_swingup_balance')
    model.save_replay_buffer("model/td3_nikki/td3_swingup_balance_replay_buffer")

# Close the environment
env.close()
