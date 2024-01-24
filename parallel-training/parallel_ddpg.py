import os
import numpy as np
from gym_env_mt import GymEnvMT
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

MAX_STEPS_PER_EPISODE = 300
SIM_STEP_FREQ_HZ = 60

# Create Isaac environment and open Sim Window
env = GymEnvMT(max_steps = MAX_STEPS_PER_EPISODE,
               sim_s_step_freq = SIM_STEP_FREQ_HZ,
               headless=False,
               experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')
spacing = 5
offsets = [[spacing, spacing, 0], [spacing, 0, 0], [spacing, -spacing, 0],
           [0, spacing, 0], [0, 0, 0], [0, -spacing, 0],
           [-spacing, spacing, 0], [-spacing, 0, 0], [-spacing, -spacing, 0]]
task_envs = env.init_tasks(offsets, backend="numpy")

def make_env(env):
    return lambda: env

env = DummyVecEnv([make_env(task_env) for task_env in task_envs])

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


model = None
try:
    model = DDPG.load("progress/ddpg")
except:
    model = DDPG("MultiInputPolicy", env,
                action_noise=action_noise,
                # learning_rate=
                # tau=,
                # gamma=,
                learning_starts=20,
                 train_freq=MAX_STEPS_PER_EPISODE * 10, # How many steps til models get updated
                batch_size=256,
                buffer_size=20000,
                verbose=1)

while (True):
    model.learn(total_timesteps=MAX_STEPS_PER_EPISODE * 20, log_interval=10)
    model.save("progress/ddpg")

print('Finished Traing')

# env.close()