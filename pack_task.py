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
DEST_BOX_POS = [2, -2, 0]

CAMERA_PATH = '/World/Camera' 
CAMERA_POS_1 = [12, -10, 1] # [3, -3, 2.5]
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

        # The UR10e has for every joint a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        # The NN outputs the change in rotation for each joint
        joint_rot_max = 190 / sim_s_step_freq
        self.action_space = spaces.Tuple(spaces=[
            spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(1,)), # Base
            spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(1,)), # Shoulder
            spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(1,)), # Elbow
            spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(1,)), # Wrist 1
            spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(1,)), # Wrist 2
            spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(1,)), # Wrist 3
            spaces.Box(low=0, high=5, shape=(1,)), # Gripper in cm of opening
        ])

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
        # self.__camera.set_focus_distance(40)

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
        quat = [dot + 1, *rot_axis]
        # orient = rot_utils.euler_angles_to_quats([0, 45, 0], degrees=True)
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
    #     return
    
    def pre_physics_step(self, actions) -> None:
        print(actions)
        joint_rots = self.robot.get_joint_positions()
        joint_rots += np.array(actions[0:6])
        joint_rots.concatenate(actions[6])
        self.robot.set_joint_positions(positions=joint_rots)
        return

    def get_observations(self):
        frame = self.__camera.get_current_frame()
        # print('Camera Image Options: ', frame.keys())
        img_rgba = frame['rgba'] # = [[[r, g, b, a]]] of size IMG_RESOLUTION[0]xIMG_RESOLUTION[1]x4
        img_depth = frame['distance_to_image_plane'] # = [[depth]] of size IMG_RESOLUTION[0]xIMG_RESOLUTION[1]
        img_seg = frame['instance_id_segmentation'] # = [[img_seg]] of size IMG_RESOLUTION[0]xIMG_RESOLUTION[1]x4
        return map(lambda rows, i: map(lambda pixel, j: [*pixel[0:3], img_depth[i][j], img_seg[i][j]], np.ndenumerate(rows)), np.ndenumerate(img_rgba))

    def calculate_metrics(self) -> None:
        return 0

    def is_done(self) -> None:
        return False