import os
from skrl.envs.wrappers.torch import OmniverseIsaacGymWrapper
from skrl.trainers.torch import ParallelTrainer
from omni.isaac.gym.vec_env import TaskStopException, VecEnvMT, TrainerMT

env = VecEnvMT(headless=False,
               sim_device=0,
               enable_viewport=False,
               experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

from pack_task_easy import PackTask
task = PackTask(f"Task_name")
env.set_task(task)

env = OmniverseIsaacGymWrapper(env)
env.run(TrainerMT)

cfg = {"timesteps": 50000, "headless": False}
trainer = ParallelTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()