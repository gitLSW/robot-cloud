import os
import numpy as np
from gym_env import GymEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

MAX_STEPS_PER_EPISODE = 300
SIM_STEP_FREQ_HZ = 60

# Create Isaac environment and open Sim Window
# env = GymEnv(headless=False, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation/manual_livestream_clients.html
env = GymEnv(headless=False, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit', enable_livestream=False)

from pack_task import PackTask # Cannot be imported before Sim has started
task = PackTask(name="Pack", max_steps=MAX_STEPS_PER_EPISODE, sim_s_step_freq=SIM_STEP_FREQ_HZ)
env.set_task(task, backend="numpy", rendering_dt=1 / SIM_STEP_FREQ_HZ)
env._world.scene.add_default_ground_plane()

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, batch_size=256, buffer_size=30000)
model.learn(total_timesteps=10000, log_interval=10)

model.save("progress/ddpg")

print('Finished Traing')

# env.close()