# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# import stable baselines
import carb

try:
    from stable_baselines3 import DDPG
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install stable-baselines3 in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install stable-baselines3 "
    )
    exit()

try:
    import tensorboard
except Exception as e:
    carb.log_error(e)
    carb.log_error(
        "please install tensorboard in the current python environment or run the following to install into the builtin python environment ./python.sh -m pip install tensorboard"
    )
    exit()

# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase

env = VecEnvBase(headless=False) # Open Sim Window

# create task and register task
from pack_task import PackTask

task = PackTask(name="Pack")
env.set_task(task, backend="torch")

# create agent from stable baselines
# model = DDPG(
#     "MlpPolicy",
#     env,
#     batch_size=1000,
#     learning_rate=0.001,
#     tau=0.005,
#     gamma=0.99,
#     device="cuda:0",
#     verbose=1,
#     tensorboard_log="./pack_tensorboard",
# )
# model.learn(total_timesteps=100000)
# model.save("ddpg_pack")

while True:
    print('Hi')

# env.close()
