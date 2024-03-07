import os
import threading

import torch
import torch.nn as nn

import wandb

# import the skrl components to build the RL system
from skrl.utils import set_seed
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import ParallelTrainer
from skrl.envs.wrappers.torch import OmniverseIsaacGymWrapper
from omniisaacgymenvs.isaac_gym_env_utils import get_env_instance

name = 'DDPG_Pack'

# seed for reproducibility
seed = set_seed()

# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1024),
                                 nn.ReLU6(),
                                 nn.Linear(1024, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 512),
                                 nn.ReLU6(),
                                 nn.Linear(512, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# Load the Isaac Gym environment
headless = True # set headless to False for rendering
multi_threaded = headless
env = get_env_instance(headless=headless,
                       multi_threaded=multi_threaded, # Multithreaded doesn't work with UI open
                       experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

from omniisaacgymenvs.sim_config import SimConfig, merge
from pack_task_part_in_gripper import PackTask as Task, TASK_CFG

TASK_CFG['name'] = name
TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
if not headless:
    TASK_CFG["task"]["env"]["numEnvs"] = 25

sim_config = SimConfig(TASK_CFG)
task = Task(name=name, sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True, rendering_dt=TASK_CFG['task']['sim']['dt'])

if multi_threaded:
    env.initialize(action_queue=env.action_queue, data_queue=env.data_queue, timeout=30)

# wrap the environment
env = OmniverseIsaacGymWrapper(env)
device = env.device

# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {
    'policy': DeterministicActor(env.observation_space, env.action_space, device),
    'target_policy': DeterministicActor(env.observation_space, env.action_space, device),
    'critic': Critic(env.observation_space, env.action_space, device),
    'target_critic': Critic(env.observation_space, env.action_space, device),
}

num_envs = TASK_CFG["task"]["env"]["numEnvs"]

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
ddpg_cfg = DDPG_DEFAULT_CONFIG.copy()
ddpg_cfg = merge({
    "exploration": {
        "noise": OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
    },
    "gradient_steps": 1,
    "batch_size": 4096,
    "discount_factor": 0.99,
    "polyak": 0.005,
    "actor_learning_rate": 5e-4,
    "critic_learning_rate": 5e-4,
    "random_timesteps": 80,
    "learning_starts": 80,
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {
        "size": env.observation_space, 
        "device": device
    },
    "experiment": {
        "directory": "progress", # experiment's parent directory
        "experiment_name": name, # experiment name
        "write_interval": 200, # TensorBoard writing interval (iterations)
        "checkpoint_interval": 1000, # interval for checkpoints (iterations)
        "store_separately": False, # whether to store checkpoints separately
        "wandb": True, # whether to use Weights & Biases
        "wandb_kwargs": {} # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}, ddpg_cfg)

run = wandb.init(
    project=name,
    config=ddpg_cfg,
    sync_tensorboard=True, # auto-upload sb3's tensorboard metrics
    # monitor_gym=True, # auto-upload the videos of agents playing the game
    # save_code=True, # optional
)

# instantiate a memory as experience replay
memory = RandomMemory(memory_size=100_000, num_envs=num_envs, device=device)
agent = DDPG(models=models,
             memory=memory,
             cfg=ddpg_cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

try:
    agent.load("./progress/DDPG_Pack/checkpoints/best_agent.pt")
except FileNotFoundError:
    print('Cloud not load agent. Created new Agent !')

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50_000_000 // num_envs, "headless": headless}
trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent)

if multi_threaded:
    # start training in a separate thread
    threading.Thread(target=trainer.train).start()
    env.run(trainer=None) # The TraimerMT can be None, cause it is only used to stop the Sim
else:
    trainer.train()