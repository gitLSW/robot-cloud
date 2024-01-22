import os
import numpy as np
from gym_env_mt import GymEnvMT

MAX_STEPS_PER_EPISODE = 500

# Create Isaac environment and open Sim Window
env = GymEnvMT(max_steps = MAX_STEPS_PER_EPISODE,
               sim_s_step_freq = 60,
               headless=False,
               experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')
spacing = 5
offsets = [[spacing, spacing, 0], [spacing, 0, 0], [spacing, -spacing, 0],
           [0, spacing, 0], [0, 0, 0], [0, -spacing, 0],
           [-spacing, spacing, 0], [-spacing, 0, 0], [-spacing, -spacing, 0]]
task_envs = env.init_tasks(offsets, backend="numpy")

while True:
    for task in task_envs:
        task.step(np.ones(7))