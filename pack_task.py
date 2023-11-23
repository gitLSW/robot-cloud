# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import math
import numpy as np
import torch

from gymnasium import spaces

from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.importer.urdf")
enable_extension("omni.isaac.universal_robots")
# from omni.importer.urdf import _urdf
from omni.isaac.universal_robots.ur10 import UR10
# from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController

# from omni.isaac.core.prims import XFormPrim
# from omni.isaac.core.articulations import ArticulationView, Articulation
# from omni.isaac.core.prims.xform_prim import XFormPrim

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.gym.tasks.rl_task import RLTaskInterface
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view

ROBOT_PATH = '/World/UR10e'
ENV_PATH = "/World/Warehouse"
START_TABLE_PATH = "/World/StartTable"
DESTINATION_BOX_PATH = "/World/DestinationBox"

ROBOT_POS = [5.45, 3, 0]

class PackTask(BaseTask):
    def __init__(self, name, offset=None) -> None:
        # values used for defining RL buffers
        self._num_observations = 4
        self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1

        # set the action and observation space for RL
        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0, np.ones(self._num_actions, dtype=np.float32) * 1.0
        )
        
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)

        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        assets_root_path = get_assets_root_path()

        # create_prim(prim_path=ENV_PATH, prim_type="Xform", position=[0, 0, 0])
        # add_reference_to_stage(assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd", ROBOT_PATH)
        
        scene.add_default_ground_plane()

        create_prim(prim_path=START_TABLE_PATH, prim_type="Xform",
                    position=[5.36, 2.29, 0],
                    scale=[0.5, 1, 0.5])
        add_reference_to_stage(assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd", START_TABLE_PATH)

        create_prim(prim_path=DESTINATION_BOX_PATH, prim_type="Xform", position=[5, 4.43, 0])
        add_reference_to_stage(assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd", DESTINATION_BOX_PATH)
        # self._box = XFormPrim(prim_path=DESTINATION_BOX_PATH)

        self.robot = UR10(prim_path=ROBOT_PATH, name='UR10e', usd_path=os.getcwd() + '/robot/ur10e.usd', position=ROBOT_POS, attach_gripper=True)
        # self.robot.set_joints_default_state(positions=[-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        self.robot.set_joint_positions(positions=[-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

        # set default camera viewport position and target
        self.reset_cameras()


    def reset_cameras(self):
        set_camera_view(eye=[7, 9, 3], target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        return

    def reset(self, env_ids=None):
        # self.robot.initialize()

        # self.robot.set_joint_positions(positions=[-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        # print(self.robot.get_joints_default_state())
        # print(self.robot.get_joint_positions())
        
        return

    # def pre_physics_step(self, actions) -> None:
    #     reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
    #     if len(reset_env_ids) > 0:
    #         self.reset(reset_env_ids)

    #     actions = torch.tensor(actions)

    #     forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
    #     forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

    #     indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
    #     self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self):
        # self.robot.end_effector
        # self.robot.gripper
        return []

    def calculate_metrics(self) -> None:
        return 0

    def is_done(self) -> None:
        return False