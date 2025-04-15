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
parser.add_argument('--load', type=str, default=True, help='Load a pre-trained model (Yes/No)', choices=['Yes', 'No'])

# Path to save the trained model
parser.add_argument('--save_model_as', type=str, default='td3_quad_reward', help='Save the model with name')

# Path to save the trained model tensorboard logs
parser.add_argument('--tensorboard_log', type=str, default=None, help='Save the tensorboard log to model/tb_date')

# Number of environments for evaluation
parser.add_argument('--num_envs', type=int, default=10, help='Number of randomized envs for evaluation')

# Total timesteps to train the model
parser.add_argument('--train_timesteps', type=int, default=None, help='Number of timesteps for training (set to None to skip training)')

# Maximum steps allowed per episode
parser.add_argument('--max_episode_step', type=int, default=6000, help='Maximum steps allowed per episode.')

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
num_envs = args.num_envs
max_episode_step = args.max_episode_step
save_model_as = args.save_model_as
seed = args.seed
tb_log = args.tensorboard_log


####### begin domain randomization #######
set_random_seed(seed)

# Parameters of environment as dict
# keys name of the parameters in env
# items range of parameters to be uniformly chosen from
params = { 
        "Rm": (2.6 * 0.95, 2.6 * 1.05), # motor armature resistance manual nominal highlighted +/- 12%
        "r_mp": (6.35e-3 * 0.95, 6.35 * 1.05), # motor pinion radius set to +/- 5%
        # "Kt": (3.44e-4, 0.014102), # motor torque constant manual extrapolated from 0 to 12V
        # "Km": (3.46e-4, 0.014037), # Back EMF constant extrapolated from 0 to 12V
        "Kt": (0.007683 * 0.95, 0.007683 * 1.05), # motor torque constant manual set to +/- 5%
        "Km": (0.007678 * 0.95, 0.007678 * 1.05), # Back EMF constant set to +/- 5%
        "Jm": (3.40e-7, 5.58e-7), # rotor moment of inertia manual fitted quadratic extrapolated from 0 to 12V
        "Bp": (0.0024 * 0.95, 0.0024 * 1.05), # equivalent viscous damping at pendulum axis set to +/- 5%
        "Beq": (5.4 * 0.95, 5.4 * 1.05)  # equivalent viscous damping at motor pinion set to +/- 5%
        }

dr_schedule_config_1 = {
            'param_name': ['Jm', 'Kt', 'Km'], 
            'nominal_value': {
                    'Jm': 3.90e-7, 
                    'Kt': 0.007683, 
                    'Km': 0.007678
                    },
            'initial_margin': 0.0, 
            'final_margin': 0.05, 
            'schedule_eps': 1e2, 
            'schedule_type': 'linear'
            }

dr_schedule_config_2 = {
            'schedules': {
                    'Jm': {'initial': 0, 'final': 0.1, 'eps': 5e3}, 
                    'Kt': {'initial': 0, 'final': 0.05, 'eps': 1e3}, 
                    'Km': {'initial': 0, 'final': 0.05, 'eps': 1e3}
                    }
            }


# Register the custom CartPoleSwingUp environment with Gym

gym.register(
    id='CartPoleSwingUpRandom',
    entry_point='myCartpoleF_random:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)

# Custom Wrapper for Curriculum Domain Randomization
class CurriculumRandomizedEnv(gym.Wrapper):
    def __init__(self, env, dr_schedule_config):
        """
        Input:
            env: existing environment that has uncertainty intervals near some parameters' nominal values
            dr_schedule_config:
                - param_name: list of parameter names to schedule
                - nominal_value: dict of nominal parameter values
                - initial_margin: initial relative margin (e.g., 0.1 = 10%)
                - final_margin: target margin (e.g., 0.5 = +/- 50%)
                - schedule_eps: number of steps to reach final margin
                - schedule_type: 'linear' or 'exponential'
        """

        super().__init__(env)
        
        
        # validate config keys
        required_keys = {'param_name', 'nominal_value', 
                        'initial_margin', 'final_margin', 
                        'schedule_eps', 'schedule_type'}
        missing = required_keys - dr_schedule_config.keys()
        if missing: 
            raise ValueError(f"Missing keys in dr_schedule_config: {missing}")
        
        self.dr_config = dr_schedule_config
        self.total_episodes = 0
        self.curriculum_updated = False # flag for curriculum
        
        # initialize margin progression
        self.current_margin = self.dr_config['initial_margin']
        self._update_dr_bounds()
        
    def _update_dr_bounds(self):
        """ update env parameters sampled within current margins """
        new_param_range = {}
        for param in self.dr_config['param_name']:
        
            nominal = self.dr_config['nominal_value'][param]
            delta = nominal * self.current_margin # e.g., if current_margin is 0.01, then deviate 1% from current
            new_param_range[param] = (nominal - delta, nominal + delta)
        
        self.env.unwrapped.set_params(new_param_range)
        return new_param_range
                
        ''' randomize by uniformly sampling new parameters within the scheduled interval 
        # this part is old
        self.current_params = {
                key: np.random.uniform(low, high) for key, (low, high) in self.param_ranges.items() 
        }
        for key, (low, high) in self.current_params.items():
            if hasattr(self.env.unwrapped, key):
                setattr(self.env.unwrapped, key, (low, high))
        '''
    
    def _update_curriculum(self):
        """ update the intervals based on training progress """
        progress = min(self.total_episodes / self.dr_config['schedule_eps'], 1.0)
        
        if self.dr_config['schedule_type'] == 'linear':
            self.current_margin = self.dr_config['initial_margin'] + (
                    self.dr_config['final_margin'] - self.dr_config['initial_margin']
                    ) * progress
        elif self.dr_config['schedule_type'] == 'exponential':
            scale = np.exp( 1 * progress ) / np.exp(1) # growth rate monitor
            self.current_margin = self.dr_config['initial_margin'] + (
                    self.dr_config['final_margin'] - self.dr_config['initial_margin']
                    ) * scale
        
        # after defining self.current_margin, apply it to change the intervals around each param in env
        new_param_range = self._update_dr_bounds()
        self.current_ranges = new_param_range
        return new_param_range
    
    def reset(self, **kwargs):
        # first update curriculum BEFORE env resets
        if self.total_episodes % self.dr_config['schedule_eps'] == 0:
            new_param_range = self._update_curriculum()
            self.curriculum_updated = True
        else:
            new_param_range = self.current_ranges
            self.curriculum_updated = False
        
        # track total episodes
        self.total_episodes += 1
        
        self.reset_params, self.reset_ranges = self.env.unwrapped.set_params(new_param_range)
        
        for key, value in self.reset_params.items():
            if hasattr(self.env.unwrapped, key):
                setattr(self.env.unwrapped, key, value)
            else:
                print(f"Warning: Unknown parameter {key}")

        return self.env.reset(**kwargs)
        
    def get_current_params(self):
        return self.reset_params


# Function to create env instance with randomized parameters from sensitivity analysis
def mod_make_env(env_id, dr_config, render_mode=None):
    def _init():
        # from registered gym env call make
        # env = gym.make('CartPoleSwingUpRandom',  render_mode='human')
        env = gym.make(env_id, render_mode = render_mode)
        return CurriculumRandomizedEnv(env, dr_config)
    return _init


envs = DummyVecEnv([mod_make_env('CartPoleSwingUpRandom', dr_schedule_config_1) for _ in range(num_envs)])

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
# def linear_schedule(initial_value):
#       return lambda: progress: progress * initial_value

def lr_schedule(lr_val):
    return lambda _: lr_val

learning_rate={ "actor": lr_schedule(1e-4), "critic": lr_schedule(3e-4)} 

if load == 'Yes':
    print("Loading the pre-trained model...")
    model = TD3.load(path= save_model_as, env=envs)
    model.load_replay_buffer(save_model_as + "_replay_buffer")
else:
    # Add noise to actions for exploration during training
    action_noise = NormalActionNoise(
        mean=np.zeros(envs.action_space.shape),
        sigma=0.08 * np.ones(envs.action_space.shape)
    )

    # Initialize a new TD3 model with a custom neural network architecture
    
    # Create the TD3 model with the custom policy
    model = TD3(
        # TD3Policy(
        #     optimizer_kwargs={
        #         "weight_decay": 1e-5, 
        #         "eps": 1e-6}
        #         ),
        TD3Policy, 
        envs,
        #gradient_clip=1.0 # clips grad norm to 1.0 prevent gradient explosion by scaling
        policy_kwargs=dict(
                        net_arch=dict(pi=[128, 256, 36], qf=[256,512,36])
                        ),
        # learning_rate=learning_rate, 
        action_noise=action_noise, 
        verbose=1, 
        tensorboard_log = tb_log,
        learning_starts=200,
        # train_freq=(2,"episode"), # training frequency defaults 1
        policy_delay=4, # policy updates every 4 training updates of Q-value
        batch_size=512, 
        buffer_size=1200000
        # tau = 0.04
    )

# custom callback to render during training
from stable_baselines3.common.callbacks import BaseCallback

class ParamLogger(BaseCallback):
    """ Keep track of domain randomization at specific ep intervals """
    def __init__(self, log_freq = 500): # log interval smaller than curriculum update interval
        super().__init__()
        self.log_freq = log_freq # log env parameter every log_freq number of steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 or self._curriculum_updated():
            env = self.training_env.envs[0]
            # progress = env.total_episodes / env.dr_config['']
            
            # log curriculum metrics
            self.logger.record("curriculum/current_margin", env.current_margin)
            params = env.get_current_params()
            for param, val in params.items():
                self.logger.record(f"params/{param}", val)
            # self._log_param_stats()
        # Additional log on curriculum updats if detectable
        # if dones[0]:
        #     self.render_env.envs[0].reset()
        return True
    
    def _curriculum_updated(self):
        """ detect if any env's curriculum has changed this episode """
        return any( env.total_episodes % env.dr_config['schedule_eps'] == 0 for env in self.training_env.envs )
        
# render_env = gym.make('CartPoleSwingUpRandom', render_mode = 'human')
# render_env = RandomizedEnv(render_env, params)
# render_env = envs
render_env = DummyVecEnv([mod_make_env('CartPoleSwingUpRandom', dr_schedule_config_1, render_mode='rgb_array') for _ in range(int(num_envs / num_envs))])

# Train the model if the user specifies a training duration
if train_timesteps is not None:
    print('--------------Training the Model--------------')
    # print("Training timesteps so far:", model.num_timesteps)
    if load == 'Yes':
        model.learn(total_timesteps=train_timesteps, 
                    tb_log_name= "another_first_1m_steps",
                    reset_num_timesteps=False,
                    callback = ParamLogger(log_freq=500))
        model.learn(total_timesteps=train_timesteps,
                    tb_log_name= "another_second_1m_steps",
                    reset_num_timesteps=False,
                    callback = ParamLogger(log_freq=500))
        model.learn(total_timesteps=train_timesteps,
                    tb_log_name= "another_third_1m_steps",
                    reset_num_timesteps=False,
                    callback = ParamLogger(log_freq=500))
    elif load == 'No':
        model.learn(total_timesteps=train_timesteps, 
                    tb_log_name= "first_1m_steps",
                    callback = ParamLogger(log_freq=500))
        model.learn(total_timesteps=train_timesteps,
                    tb_log_name= "second_1m_steps",
                    reset_num_timesteps=False,
                    callback = ParamLogger(log_freq=500))
        model.learn(total_timesteps=train_timesteps,
                    tb_log_name= "third_1m_steps",
                    reset_num_timesteps=False,
                    callback = ParamLogger(log_freq=500))

# Save the trained model if a save path is specified
if save_model_as is not None:
    print(f"Saving the model to %s", save_model_as)
    model.save(save_model_as)
    model.save_replay_buffer(save_model_as + '_replay_buffer')

# Close the environment
render_env.close()
