import argparse
import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3.common.utils import set_random_seed

# Setting up the argument parser for command-line inputs
parser = argparse.ArgumentParser()

# Device to run the code on (CPU or GPU)
parser.add_argument('--device', type=str, default='cpu', help='Specify device: cpu or cuda')

# Whether to load a pre-trained model
parser.add_argument('--load', type=str, default='True', help='Load a pre-trained model (True/False)')


# Path to save the trained model
parser.add_argument('--save', default=True, help='Save the model or not')

# Number of episodes for evaluation
parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')

# Total timesteps to train the model
parser.add_argument('--train_timesteps', type=int, default=None, help='Number of timesteps for training (set to None to skip training)')

# Maximum steps allowed per episode
parser.add_argument('--max_episode_step', type=int, default=None, help='Maximum steps allowed per episode.')

parser.add_argument('--seed', type=int, default=4, help='Random seed to guarantee reproducibility')
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
    model = TD3.load(path='model/td3/td3_swingup_balance', env=env)
    model.load_replay_buffer("model/td3/td3_swingup_balance_replay_buffer")
else:
    # Add noise to actions for exploration during training
    action_noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=0.1 * np.ones(env.action_space.shape)
    )

    # Initialize a new TD3 model with a custom neural network architecture
    model = TD3(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[128, 256, 32]),
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
        while not done:
            # Predict the action
            action, _ = model.predict(obs, deterministic=False)
            # Take the action and observe the result
            obs, reward, done, info = env.step(action)
            # Add the reward to the total
            total_reward += reward
            # Render the environment
            env.render()
        print(f'Episode: {episode + 1} | Total Reward: {total_reward}')

# Save the trained model if a save path is specified
if save:
    print(f"Saving the model")
    model.save('model/td3/td3_swingup_balance')
    model.save_replay_buffer("model/td3/td3_swingup_balance_replay_buffer")

# Close the environment
env.close()
