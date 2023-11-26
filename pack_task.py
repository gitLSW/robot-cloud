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
import glm
import torch

from gymnasium import spaces

from scipy.spatial.transform import Rotation as R

# from camera import VideoFeed
from omni.isaac.sensor import Camera
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
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
import omni.isaac.core.utils.numpy.rotations as rot_utils

ENV_PATH = "/World/Env"

ROBOT_PATH = '/World/UR10e'
ROBOT_POS = [0, 0, 0]
# 5.45, 3, 0
START_TABLE_PATH = "/World/StartTable"
START_TABLE_POS = [0.36, 1.29, 0]

DEST_BOX_PATH = "/World/DestinationBox"
DEST_BOX_POS = [0, -0.5, 0]

CAMERA_PATH = '/World/Camera'
CAMERA_POS_1 = [10, -15, 7]# [3, -3, 2.5]
IMG_RESOLUTION = (512, 512)

class PackTask(BaseTask):
    """
    This class sets up a scene and calls a RL Policy, then evaluates the behaivior with rewards
    Args:
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        sim_s_step_freq (int): The amount of simulation steps within a SIMULATED second.
    """

    def __init__(self, name, offset=None, sim_s_step_freq: int = 60) -> None:
        # self._num_observations = 4
        # self._num_actions = 1
        # self._device = "cpu"
        self.num_envs = 1
        self.sim_s_step_freq = sim_s_step_freq

        # The NN will see the Robot via a single video feed that can run from one of two camera positions
        # The NN will receive this feed in rgb, depth and image segmented to highlight objects of interest
        # We need 7 image dimensions: for rgb (=>3), depth (black/white=>1), image_segmentation (=rgb=>3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(IMG_RESOLUTION[0], IMG_RESOLUTION[1], 7))

        # The NN outputs the rotations for each joint
        self.action_space = spaces.Box(low=0, high=255, shape=(IMG_RESOLUTION[0], IMG_RESOLUTION[1], 7))
        # self.action_space = spaces.Sequence(space=[
        #     spaces.Box(low=-360, high=360, shape=(1,)), # Base
        #     spaces.Box(low=-360, high=360, shape=(1,)), # Shoulder
        #     spaces.Box(low=-360, high=360, shape=(1,)), # Elbow
        #     spaces.Box(low=-360, high=360, shape=(1,)), # Wrist 1
        #     spaces.Box(low=-360, high=360, shape=(1,)), # Wrist 2
        #     spaces.Box(low=-360, high=360, shape=(1,)), # Wrist 3
        # ])

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)

        local_assets = os.getcwd() + '/assets'

        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        # assets_root_path = get_assets_root_path()

        # create_prim(prim_path=ENV_PATH, prim_type="Xform", position=[0, 0, 0])
        # warehouse_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        # add_reference_to_stage(warehouse_path, ROBOT_PATH)
        
        scene.add_default_ground_plane()

        create_prim(prim_path=START_TABLE_PATH, prim_type="Xform",
                    position=START_TABLE_POS,
                    scale=[0.5, 1, 0.5])
        # table_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        add_reference_to_stage(local_assets + '/table_low.usd', START_TABLE_PATH)

        create_prim(prim_path=DEST_BOX_PATH, prim_type="Xform", position=DEST_BOX_POS)
        # box_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd"
        add_reference_to_stage(local_assets + '/SM_CardBoxA_02.usd', DEST_BOX_PATH)
        # self._box = XFormPrim(prim_path=DEST_BOX_PATH)

        self.robot = UR10(prim_path=ROBOT_PATH, name='UR10e', usd_path=local_assets + '/ur10e.usd', position=ROBOT_POS, attach_gripper=True)
        # self.robot.set_joints_default_state(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))
        
        self.__camera = Camera(
            prim_path=CAMERA_PATH,
            frequency=20,
            resolution=IMG_RESOLUTION,
            position=torch.tensor(CAMERA_POS_1),
            # orientation=torch.tensor([1, 0, 0, 0])
        )

        self.__moveCamera(position=CAMERA_POS_1, target=ROBOT_POS)

        viewport = get_active_viewport()
        viewport.set_active_camera(CAMERA_PATH)

        # set_camera_view(eye=[7, 9, 3], target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")
        
    def __moveCamera(self, position, target):
        pos = np.array(position)
        target = np.array(target)
        dir_vec = target - pos
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        forwards = np.array([1, 0, 0])
        rot_axis = np.cross(forwards, dir_vec)
        dot = np.dot(forwards, dir_vec)

        # orient = rot_utils.euler_angles_to_quats([0, 45, 0], degrees=True)
        quat = [dot + 1, rot_axis[0], rot_axis[1], rot_axis[2]]
        print(quat)
        self.__camera.set_world_pose(position=position, orientation=quat)
        return

    def post_reset(self):
        return

    def reset(self, env_ids=None):
        self.robot.initialize()
        self.__camera.initialize()
        self.__camera.add_distance_to_image_plane_to_frame() # depth cam
        self.__camera.add_instance_id_segmentation_to_frame() # simulated segmentation NN
        self.robot.set_joint_positions(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))
        return

    # def pre_step(self, time_step_index: int, simulation_time: float) -> None:
    #     """called before stepping the physics simulation.

    #     Args:
    #         time_step_index (int): [description]
    #         simulation_time (float): [description]
    #     """
    #     return
    
    def pre_physics_step(self, actions) -> None:
        # print('Camera Image Options: ', self.camera.get_current_frame().keys())
        frame = self.__camera.get_current_frame()
        img_rgba = frame['rgba']
        img_depth = frame['distance_to_image_plane']
        img_seg = frame['instance_id_segmentation']

    #     reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
    #     if len(reset_env_ids) > 0:
    #         self.reset(reset_env_ids)

    #     actions = torch.tensor(actions)

    #     forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
    #     forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

    #     indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
    #     self._cartpoles.set_joint_efforts(forces, indices=indices)
        return

    def get_observations(self):
        # self.robot.end_effector
        # self.robot.gripper
        return []

    def calculate_metrics(self) -> None:
        return 0

    def is_done(self) -> None:
        return False