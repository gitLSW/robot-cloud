import os
import threading

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import ParallelTrainer
from skrl.utils import set_seed
from skrl.envs.wrappers.torch import OmniverseIsaacGymWrapper
from omni.isaac.gym.vec_env import TaskStopException, VecEnvMT


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the Isaac Gym environment
# tHE vEC_ENV_mt FILLS THE QUUES with data

from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.omniverse_isaacgym_utils import get_env_instance

# get environment instance
env = get_env_instance(headless=True)

# parse sim configuration
from sim_config import SimConfig
sim_config = SimConfig({"test": False,
                        "device_id": 0,
                        "headless": True,
                        "multi_gpu": False,
                        "sim_device": "gpu",
                        "enable_livestream": False,
                        "task": {"name": "CustomTask",
                                    "physics_engine": "physx",
                                    "env": {"numEnvs": 512,
                                            "envSpacing": 1.5,
                                            "enableDebugVis": False,
                                            "clipObservations": 1000.0,
                                            "clipActions": 1.0,
                                            "controlFrequencyInv": 4},
                                    "sim": {"dt": 0.0083,  # 1 / 120
                                            "use_gpu_pipeline": True,
                                            "gravity": [0.0, 0.0, -9.81],
                                            "add_ground_plane": True,
                                            "use_flatcache": True,
                                            "enable_scene_query_support": False,
                                            "enable_cameras": False,
                                            "default_physics_material": {"static_friction": 1.0,
                                                                        "dynamic_friction": 1.0,
                                                                        "restitution": 0.0},
                                            "physx": {"worker_thread_count": 4,
                                                    "solver_type": 1,
                                                    "use_gpu": True,
                                                    "solver_position_iteration_count": 4,
                                                    "solver_velocity_iteration_count": 1,
                                                    "contact_offset": 0.005,
                                                    "rest_offset": 0.0,
                                                    "bounce_threshold_velocity": 0.2,
                                                    "friction_offset_threshold": 0.04,
                                                    "friction_correlation_distance": 0.025,
                                                    "enable_sleeping": True,
                                                    "enable_stabilization": True,
                                                    "max_depenetration_velocity": 1000.0,
                                                    "gpu_max_rigid_contact_count": 524288,
                                                    "gpu_max_rigid_patch_count": 33554432,
                                                    "gpu_found_lost_pairs_capacity": 524288,
                                                    "gpu_found_lost_aggregate_pairs_capacity": 262144,
                                                    "gpu_total_aggregate_pairs_capacity": 1048576,
                                                    "gpu_max_soft_body_contacts": 1048576,
                                                    "gpu_max_particle_contacts": 1048576,
                                                    "gpu_heap_capacity": 33554432,
                                                    "gpu_temp_buffer_capacity": 16777216,
                                                    "gpu_max_num_partitions": 8}}}})

# import and setup custom task
from pack_task import PackTask
task = PackTask(fame="CustomTask", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

# wrap the environment
env = OmniverseIsaacGymWrapper(env)

device = env.device

# instantiate a memory as experience replay
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/Ant"

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)


# configure and instantiate the RL trainer
cfg = {"timesteps": 50000, "headless": False}

# This trainer grabs the data and trains the model
trainer = ParallelTrainer(env=env, agents=agents, cfg=cfg)

threading.Thread(target=trainer.train).start()
# trainer.eval()

# The TraimerMT can be None, cause it is only used to stop the Sim
env.run(trainer=None)
