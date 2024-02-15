import os
import threading

from skrl.envs.wrappers.torch import OmniverseIsaacGymWrapper
from skrl.trainers.torch import ParallelTrainer

from omni.isaac.gym.vec_env import TaskStopException, VecEnvMT

# tHE vEC_ENV_mt FILLS THE QUUES with data
env = VecEnvMT(headless=False,
               sim_device=0,
               enable_viewport=False,
               experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

from pack_task_easy import PackTask
task = PackTask(f"Task_name")
env.set_task(task)

env = OmniverseIsaacGymWrapper(env)

cfg = {"timesteps": 50000, "headless": False}

# This trainer grabs the data and trains the model
trainer = ParallelTrainer(env=env, agents=agents, cfg=cfg)

threading.Thread(target=trainer.train).start()
# trainer.eval()

# The TraimerMT can be None, cause it is only used to stop the Sim
env.run(trainer=None)
