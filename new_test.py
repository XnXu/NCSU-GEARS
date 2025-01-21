from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy



CartPoleSwingUp = gym.register(id='CartPoleSwingUp',
  #                           entry_point='gymnasium.envs.classic_control.myCartpoleF_SwingUp:CartPoleSwingUp',
                             entry_point='myCartpoleF_SwingUp:CartPoleSwingUp',
                             reward_threshold = 0,
                                #max_episode_steps=1000
                               )

env = DummyVecEnv([lambda: gym.make('CartPoleSwingUp', render_mode='human')])



action_noise = NormalActionNoise(mean = np.zeros(env.action_space.shape), sigma = 0.1 * np.ones(env.action_space.shape))
model = DDPG.load('model/1124_cp',env=env)
#model = DDPG('MlpPolicy', env, policy_kwargs = dict(net_arch=[36, 48, 16]), action_noise = action_noise)
#model.learn(total_timesteps = 100000)


action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape))

# Load the pre-trained model and assign the environment
'''model = DDPG.load("model/ddpg_cartpole_model1", env=env)
model.learn(total_timesteps=100000)
model.save("model/ddpg_cartpole_model1")'''


# test
for _ in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render()
#model.save('1124')
env.close()
