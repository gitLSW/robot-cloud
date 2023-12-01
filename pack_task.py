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

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.gym.tasks.rl_task import RLTaskInterface
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
import omni.isaac.core.objects as objs
import omni.isaac.core.utils.numpy.rotations as rot_utils

ENV_PATH = "/World/Env"

ROBOT_PATH = '/World/UR10e'
ROBOT_POS = [0, 0, 0]
# 5.45, 3, 0
START_TABLE_PATH = "/World/StartTable"
START_TABLE_POS = [0.36, 1.29, 0]

DEST_BOX_PATH = "/World/DestinationBox"
DEST_BOX_POS = [2, -2, 0]

PARTS_PATH = '/World/Parts'
PARTS_SOURCE = np.array(START_TABLE_POS) + np.array([0, 0, 3])

CAMERA_PATH = '/World/Camera' 
CAMERA_POS_START = [3, -3, 2.5] 
CAMERA_POS_DEST = [4, 4, 2.5]
IMG_RESOLUTION = (512, 512)

class PackTask(BaseTask):
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
        self.sim_s_step_freq = sim_s_step_freq

        # The NN will see the Robot via a single video feed that can run from one of two camera positions
        # The NN will receive this feed in rgb, depth and image segmented to highlight objects of interest
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=1, shape=(IMG_RESOLUTION[0], IMG_RESOLUTION[1], 3)), # rgb => 3 Dim (Normalized from 0-255 to 0-1)
            'depth': spaces.Box(low=0, high=1, shape=(IMG_RESOLUTION[0], IMG_RESOLUTION[1], 1)), # depth in m => 1 Dim (Normalized from 0-2m to 0-1)
            'seg': spaces.Discrete(4) # segmentation using one-hot encoding in 4 Groups
        })

        # The NN outputs the change in rotation for each joint
        joint_rot_max = 190 / sim_s_step_freq
        self.action_space = spaces.Dict({
            'joint_rots': spaces.Box(low=-joint_rot_max, high=joint_rot_max, shape=(6,), dtype=float), # Joint rotations
            'gripper_open': spaces.Discrete(2) # Gripper is open = 1, else 0
        })

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
                    semantic_label='StartTable',
                    semantic_type='Start',
                    position=START_TABLE_POS,
                    scale=[0.5, 1, 0.5])
        # table_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        add_reference_to_stage(local_assets + '/table_low.usd', START_TABLE_PATH)

        create_prim(prim_path=DEST_BOX_PATH, prim_type="Xform",
                    semantic_label='DestBox',
                    semantic_type='Dest',
                    position=DEST_BOX_POS)
        # box_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd"
        add_reference_to_stage(local_assets + '/SM_CardBoxA_02.usd', DEST_BOX_PATH)
        # self._box = XFormPrim(prim_path=DEST_BOX_PATH)

        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        self.robot = UR10(prim_path=ROBOT_PATH, name='UR16e', usd_path=local_assets + '/ur16e.usd', position=ROBOT_POS, attach_gripper=True)
        # self.robot.set_joints_default_state(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))

        for i in range(5):
            scene.add(objs.DynamicCuboid(prim_path=f'{PARTS_PATH}/Part_{i}', name=f'Part_{i}',
                                         position=PARTS_SOURCE,
                                         scale=[0.1, 0.1, 0.1]))

        self.__camera = Camera(
            prim_path=CAMERA_PATH,
            frequency=20,
            resolution=IMG_RESOLUTION,
            # position=torch.tensor(CAMERA_POS_START),
            # orientation=torch.tensor([1, 0, 0, 0])
        )
        # self.__camera.set_focus_distance(40)

        self.__moveCamera(position=CAMERA_POS_START, target=ROBOT_POS)

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

        # From ~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/kit/exts/omni.kit.viewport.utility/omni/kit/viewport/utility/camera_state.py
        # parent_xform = usd_camera.ComputeParentToWorldTransform(self.__time)
        # iparent_xform = parent_xform.GetInverse()
        # initial_local_xform = world_xform * iparent_xform
        # pos_in_parent = iparent_xform.Transform(world_position)
        #         
        # cam_up = self.get_world_camera_up(cam_prim.GetStage())
        # coi_in_parent = iparent_xform.Transform(world_target)
        # new_local_transform = Gf.Matrix4d(1).SetLookAt(pos_in_parent, coi_in_parent, cam_up).GetInverse()
        # new_local_coi = (new_local_transform * parent_xform).GetInverse().Transform(world_target)

        return

    def reset(self, env_ids=None):
        self.robot.initialize()
        self.__camera.initialize()
        self.__camera.add_distance_to_image_plane_to_frame() # depth cam
        self.__camera.add_instance_id_segmentation_to_frame() # simulated segmentation NN
        self.robot.set_joint_positions(positions=np.array([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))
        return

    def get_observations(self):
        frame = self.__camera.get_current_frame()
        # print('Camera Image Options: ', frame.keys())
        img_rgba = frame['rgba'] # = [[[ 0 <= r, g, b, a <= 255 ]]] of size IMG_RESOLUTION[0]xIMG_RESOLUTION[1]x4 
        img_depth = frame['distance_to_image_plane'] # = [[ 0 <= depth <= infinity ]] of size IMG_RESOLUTION[0]xIMG_RESOLUTION[1]
        img_seg_dict = frame['instance_id_segmentation']

        one_hot_img_seg = np.zeros(IMG_RESOLUTION)
        if img_seg_dict:
            img_seg_info_dict = img_seg_dict['info']
            img_seg = img_seg_dict['data'] # = [[ 0 <= img_seg <= 0 ]] of size IMG_RESOLUTION[0]xIMG_RESOLUTION[1]x4

            def seg_label_filter(pixel_label):
                pixel_obj_path = img_seg[pixel_label]
                if pixel_obj_path == ROBOT_PATH:
                    return 1
                elif pixel_obj_path == PARTS_PATH:
                    return 2
                elif pixel_obj_path == DEST_BOX_PATH:
                    return 3
                else:
                    return 0
                
            one_hot_img_seg = map(lambda rows: map(seg_label_filter, rows), img_seg).reshape(-1)
            one_hot_img_seg = np.eye(4)[one_hot_img_seg] # We have 4 seg_obj_classes

        # TODO: Check if this is correct
        print(one_hot_img_seg)

        return (
            map(lambda rows: map(lambda pixel: [np.array([*pixel[0:3]]) / 255], rows), img_rgba),
            map(lambda rows: map(lambda pixel: pixel / 2 if pixel < 2 else 1, rows), img_depth) if img_depth else np.ones(IMG_RESOLUTION), # Normalized between 0-1 using max distance of 2m
            one_hot_img_seg
        )
    
    def pre_physics_step(self, actions) -> None:
        # print(actions)
        # Rotate Joints
        joint_rots = self.robot.get_joint_positions()
        joint_rots += np.array(actions[0:6]) * math.pi / 180 # NN uses deg and isaac rads
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
        return

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
        #     self.__moveCamera(new_cam_pose, ROBOT_POS)
        return -gripper_to_dest

    def is_done(self) -> None:
        return False