import os
import math
import numpy as np
from gymnasium import spaces

from omni.isaac.sensor import Camera
from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.importer.urdf")
enable_extension("omni.isaac.universal_robots")
# from omni.importer.urdf import _urdf
from omni.isaac.universal_robots.ur10 import UR10
# from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController

from omni.isaac.core.prims import XFormPrim, RigidPrim, GeometryPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.gym.tasks.rl_task import RLTaskInterface
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
import omni.isaac.core.objects as objs
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.rotations import lookat_to_quatf, gf_quat_to_np_array

from pxr import Gf

ENV_PATH = "/World/Env"

ROBOT_PATH = '/World/UR10e'
ROBOT_POS = np.array([0, 0, 0])
# 5.45, 3, 0
START_TABLE_PATH = "/World/StartTable"
START_TABLE_POS = np.array([0.36, 1.29, 0])

DEST_BOX_PATH = "/World/DestinationBox"
DEST_BOX_POS = np.array([0.6, -0.64, 0])

PARTS_PATH = '/World/Parts'
PARTS_SOURCE = START_TABLE_POS + np.array([0, 0, 3])

CAMERA_PATH = '/World/Camera' 
CAMERA_POS_START = np.array([3, -3, 2.5])
CAMERA_POS_DEST = np.array([4, 4, 2.5])
IMG_RESOLUTION = (512, 512)

class PackTask(BaseTask):
    parts = []

    """
    This class sets up a scene and calls a RL Policy, then evaluates the behaivior with rewards
    Args:
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        sim_s_step_freq (int): The amount of simulation steps within a SIMULATED second.
    """
    def __init__(self, name, offset=None, sim_s_step_freq: int = 60) -> None:
        # self._num_observations = 1
        # self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1
        self.__joint_rot_max = (190.0 * math.pi / 180) / sim_s_step_freq

        # The NN will see the Robot via a single video feed that can run from one of two camera positions
        # The NN will receive this feed in rgb, depth and image segmented to highlight objects of interest
        self.observation_space = spaces.Box(low=0, high=1, shape=(*IMG_RESOLUTION, 7))
        # self.observation_space = spaces.Dict({
        #     'rgb': spaces.Box(low=0, high=1, shape=(IMG_RESOLUTION[0], IMG_RESOLUTION[1], 3)), # rgb => 3 Dim (Normalized from 0-255 to 0-1)
        #     'depth': spaces.Box(low=0, high=1, shape=(IMG_RESOLUTION[0], IMG_RESOLUTION[1], 1)), # depth in m => 1 Dim (Normalized from 0-2m to 0-1)
        #     'seg': spaces.Discrete(4) # segmentation using one-hot encoding in 4 Groups
        # })

        # The NN outputs the change in rotation for each joint as a fraction of the max rot speed per timestep (=__joint_rot_max)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=float)
        # spaces.Dict({
        #     'joint_rots': spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(6,), dtype=float), # Joint rotations
        #     'gripper_open': spaces.Discrete(2) # Gripper is open = 1, else 0
        # })

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)

        local_assets = os.getcwd() + '/assets'

        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        # assets_root_path = get_assets_root_path()

        # _ = XFormPrim(prim_path=ENV_PATH, position=-np.array([5, 4.5, 0]))
        # warehouse_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        # add_reference_to_stage(warehouse_path, ENV_PATH)
        
        scene.add_default_ground_plane()

        # table_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        table_path = local_assets + '/table_low.usd'
        self.table = create_prim(prim_path=START_TABLE_PATH, usd_path=table_path, position=START_TABLE_POS, scale=[0.5, 1, 0.5])
        add_reference_to_stage(table_path, START_TABLE_PATH)
        # self.table = RigidPrim(prim_path=START_TABLE_PATH, position=START_TABLE_POS)
        # self.table.enable_rigid_body_physics()
        # scene.add(self.table)

        # box_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd"
        box_path = local_assets + '/SM_CardBoxA_02.usd'
        self.box = XFormPrim(prim_path=DEST_BOX_PATH, position=DEST_BOX_POS)
        add_reference_to_stage(box_path, DEST_BOX_PATH)

        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        self.robot = UR10(prim_path=ROBOT_PATH, name='UR16e', usd_path=local_assets + '/ur16e.usd', position=ROBOT_POS, attach_gripper=True)
        # self.robot.set_joints_default_state(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))

        for i in range(5):
            part = objs.DynamicCuboid(prim_path=f'{PARTS_PATH}/Part_{i}', name=f'Part_{i}',
                                         position=PARTS_SOURCE,
                                         scale=[0.1, 0.1, 0.1])
            # part.set_default_state(position=PARTS_SOURCE)
            scene.add(part)
            self.parts.append(part)


        self.__camera = Camera(
            prim_path=CAMERA_PATH,
            frequency=20,
            resolution=IMG_RESOLUTION,
            # position=torch.tensor(CAMERA_POS_START),
            # orientation=torch.tensor([1, 0, 0, 0])
        )
        # self.__camera.set_focus_distance(40)

        self.__moveCamera(position=CAMERA_POS_START, target=START_TABLE_POS)

        viewport = get_active_viewport()
        viewport.set_active_camera(CAMERA_PATH)

        # set_camera_view(eye=ROBOT_POS + np.array([1.5, 6, 1.5]), target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")

        self._move_task_objects_to_their_frame()
    
    def __moveCamera(self, position, target):
        # USD Frame flips target and position, so they have to be flipped here
        quat = gf_quat_to_np_array(lookat_to_quatf(camera=Gf.Vec3f(*target),
                                                   target=Gf.Vec3f(*position),
                                                   up=Gf.Vec3f(0, 0, 1)))
        self.__camera.set_world_pose(position=position, orientation=quat, camera_axes='usd')
        

    def reset(self):
        # self.table.initialize()
        self.box.initialize()
        self.robot.initialize()
        self.__camera.initialize()
        self.__camera.add_distance_to_image_plane_to_frame() # depth cam
        self.__camera.add_instance_id_segmentation_to_frame() # simulated segmentation NN
        self.robot.set_joint_positions(positions=np.array([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))
        # return np.zeros((*IMG_RESOLUTION, 7))

    def get_observations(self):
        frame = self.__camera.get_current_frame()

        img_rgba = frame['rgba']  # Shape: (Width, Height, 4)
        img_rgb = img_rgba[:, :, :3] / 255.0 # Remove alpha from rgba and scale between 0-1

        img_depth = frame['distance_to_image_plane']  # Shape: (Width, Height)
        if img_depth is not None:
            img_depth = np.clip(img_depth, 0, 2) / 2.0 # Clip depth at 2m and scale between 0-1
            img_depth = img_depth[:, :, np.newaxis]
            # img_depth = np.expand_dims(img_depth, axis=-1)
        else:
            img_depth = np.zeros((*IMG_RESOLUTION, 1))

        # Segmentation
        img_seg_dict = frame['instance_id_segmentation']
        one_hot_img_seg = img_seg = np.zeros((*IMG_RESOLUTION, 3))

        if img_seg_dict:
            img_seg_info_dict = img_seg_dict['info'] # Dict: [pixel label: prim path]
            img_seg = img_seg_dict['data']  # Shape: (Width, Height)
            
            # Vectorised One-Hot-Encoding
            for label, path in img_seg_info_dict.items():
                mask = (img_seg == label)
                if path == ROBOT_PATH:
                    one_hot_img_seg[:, :, 0] = mask
                elif path == PARTS_PATH:
                    one_hot_img_seg[:, :, 1] = mask
                elif path == DEST_BOX_PATH:
                    one_hot_img_seg[:, :, 2] = mask
        return np.concatenate([img_rgb, img_depth, one_hot_img_seg], axis=-1)
    
    def pre_physics_step(self, actions) -> None:
        # Rotate Joints
        joint_rots = self.robot.get_joint_positions()
        joint_rots += np.array(actions[0:6]) * self.__joint_rot_max # NN uses deg and isaac rads
        self.robot.set_joint_positions(positions=joint_rots)
        # Open or close Gripper
        gripper = self.robot.gripper
        curr_gripper_state = gripper.is_closed()
        new_gripper_state = actions[6]
        if (new_gripper_state != curr_gripper_state):
            if (new_gripper_state == 1):
                gripper.open()
            else:
                gripper.close()

    # Calculate Rewards
    def calculate_metrics(self) -> None:
        gripper = self.robot.gripper
        gripper_pos = gripper.get_world_pose()[0]
        gripper_to_start = np.linalg.norm(PARTS_SOURCE - gripper_pos)
        gripper_to_dest = np.linalg.norm(DEST_BOX_POS - gripper_pos)
        
        # Move Camera
        # curr_cam_pos = self.__camera.get_world_pose()[0]
        # new_cam_pose = CAMERA_POS_DEST if (gripper_to_dest < gripper_to_start) else CAMERA_POS_START
        # if not np.array_equal(new_cam_pose, curr_cam_pos):
        #     cam_target = DEST_BOX_POS if (gripper_to_dest < gripper_to_start) else START_TABLE_POS
        #     self.__moveCamera(new_cam_pose, cam_target)
        done = False
        return -gripper_to_dest, done, {}

    def is_done(self) -> None:
        return False