import os
import numpy as np
from gym_env_mt import GymEnvMT
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, subproc_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback


MAX_STEPS_PER_EPISODE = 300
SIM_STEP_FREQ_HZ = 60

# Create Isaac environment and open Sim Window
env = GymEnvMT(max_steps = MAX_STEPS_PER_EPISODE,
               sim_s_step_freq = SIM_STEP_FREQ_HZ,
               headless=False,
               experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')
spacing = 5
offsets = []
tasks_per_side_1 = 50
tasks_per_side_2 = 50

# USE GRIP CLONER INSTEAD !!!
# CHECK THSI FROM OmniIsaacGymEnvs RLTask:

        # self._cloner = GridCloner(spacing=self._env_spacing)
        # self._cloner.define_base_env(self.default_base_env_path)

        # stage = omni.usd.get_context().get_stage()
        # UsdGeom.Xform.Define(stage, self.default_zero_env_path)

        # if self._task_cfg["sim"].get("add_ground_plane", True):
        #     self._ground_plane_path = "/World/defaultGroundPlane"
        #     collision_filter_global_paths.append(self._ground_plane_path)
        #     scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        # prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        # self._env_pos = self._cloner.clone(
        #     source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics, copy_from_source=copy_from_source
        # )
        # self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        # if filter_collisions:
        #     self._cloner.filter_collisions(
        #         self._env.world.get_physics_context().prim_path,
        #         "/World/collisions",
        #         prim_paths,
        #         collision_filter_global_paths,
        #     )

from omni.isaac.cloner import GridCloner

NUM_ENVS = tasks_per_side_1 * tasks_per_side_2
for i in range(tasks_per_side_1):
    for j in range(tasks_per_side_2):
        offsets.append([i * spacing, j * spacing, 0])

task_envs = env.init_tasks(offsets, backend="numpy")

def make_env(env):
    return lambda: env

env = DummyVecEnv([make_env(task_env) for task_env in task_envs])

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


ddpg_config = {
    'learning_starts': 0,
    'action_noise': action_noise,
    'learning_rate': 0.001,
    'tau': 0.005,
    'gamma': 0.99,
    'learning_starts': 0,
    'train_freq': MAX_STEPS_PER_EPISODE * NUM_ENVS, # How many steps until models get updated
    'batch_size': 256,
    'buffer_size': MAX_STEPS_PER_EPISODE * NUM_ENVS,
    'verbose': 1
}

name = 'ddpg-pack'
run = wandb.init(
    project=name,
    config=ddpg_config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)
ddpg_config['tensorboard_log'] = f"./progress/runs/{run.id}"

model = None
try:
    model = DDPG.load(f"progress/{name}", env, print_system_info=True, custom_objects=ddpg_config)
    # model.set_parameters(params)
except FileNotFoundError:
    print('Failed to load model')
finally:
    model = DDPG("MultiInputPolicy", env, **ddpg_config)

while (True):
    model.learn(total_timesteps=MAX_STEPS_PER_EPISODE * 2, log_interval=NUM_ENVS, tb_log_name='DDPG',
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ))
    print('Saving model')
    model.save("progress/ddpg")

run.finish()
print('Finished Traing')

# env.close()