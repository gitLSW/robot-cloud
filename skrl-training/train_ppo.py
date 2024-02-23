import threading
import time

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.utils import set_seed
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import ParallelTrainer, SequentialTrainer
from skrl.envs.wrappers.torch.omniverse_isaacgym_envs import OmniverseIsaacGymWrapper
from omniisaacgymenvs.isaac_gym_env_utils import get_env_instance

# Seed for reproducibility
seed = set_seed()  # e.g. `set_seed(42)` for fixed seed

# Define the models (stochastic and deterministic models) for the agent using helper mixin.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


headless = True  # set headless to False for rendering
multi_threaded = headless
env = get_env_instance(headless=headless, multi_threaded=multi_threaded) # Multithreaded doesn't work with UI open

from omniisaacgymenvs.sim_config import SimConfig, merge
from franka_reach_task import ReachingFrankaTask, TASK_CFG

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 50_000 if headless else 25
TASK_CFG["task"]["env"]["controlSpace"] = "joint"  # "joint" or "cartesian"

sim_config = SimConfig(TASK_CFG)
task = ReachingFrankaTask(name="ReachingFranka", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True, rendering_dt=TASK_CFG['task']['sim']['dt'])
# task.reset()

if multi_threaded:
    env.initialize(action_queue=env.action_queue, data_queue=env.data_queue, timeout=5)

# wrap the environment
env = OmniverseIsaacGymWrapper(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 16
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 8
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 5e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 32 and 250 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 32
cfg_ppo["experiment"]["checkpoint_interval"] = 250

agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# agent.load("./runs/24-02-23_10-38-35-900063_PPO/checkpoints/best_agent.pt")

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50_000_000 // TASK_CFG["task"]["env"]["numEnvs"], "headless": headless}
trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent)

if multi_threaded:
    # start training in a separate thread
    threading.Thread(target=trainer.train).start()
    env.run(trainer=None) # The TraimerMT can be None, cause it is only used to stop the Sim
else:
    trainer.train()