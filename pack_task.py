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
from omni.physx.scripts.utils import setRigidBody, setStaticCollider, setCollider, addCollisionGroup


# MESH_APPROXIMATIONS = {
#         "none": PhysxSchema.PhysxTriangleMeshCollisionAPI,
#         "convexHull": PhysxSchema.PhysxConvexHullCollisionAPI,
#         "convexDecomposition": PhysxSchema.PhysxConvexDecompositionCollisionAPI,
#         "meshSimplification": PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI,
#         "convexMeshSimplification": PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI,
#         "boundingCube": None,
#         "boundingSphere": None,
#         "sphereFill": PhysxSchema.PhysxSphereFillCollisionAPI,
#         "sdf": PhysxSchema.PhysxSDFMeshCollisionAPI,
# }

from pxr import Gf

ENV_PATH = "/World/Env"

ROBOT_PATH = '/World/UR10e'
ROBOT_POS = np.array([0.0, 0.0, 0.0])
# 5.45, 3, 0
START_TABLE_PATH = "/World/StartTable"
START_TABLE_POS = np.array([0.36, 0.8, 0])
START_TABLE_HEIGHT = 0.6
START_TABLE_CENTER = START_TABLE_POS + np.array([0, 0, START_TABLE_HEIGHT])

DEST_BOX_PATH = "/World/DestinationBox"
DEST_BOX_POS = np.array([0, -0.65, 0])

PARTS_PATH = '/World/Parts'
PARTS_SOURCE = START_TABLE_CENTER + np.array([0, 0, 1])
# NUM_PARTS = 5

CAMERA_PATH = '/World/Camera' 
CAMERA_POS_START = np.array([-2, 2, 2.5])
CAMERA_POS_DEST = np.array([2, -2, 2.5])
IMG_RESOLUTION = (512, 512)

class PackTask(BaseTask):
    part = None

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
        # Robot turning ange of max speed is 191deg/s
        self.__joint_rot_max = (191.0 * math.pi / 180) / sim_s_step_freq

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
        self.table = XFormPrim(prim_path=START_TABLE_PATH, position=START_TABLE_POS, scale=[0.5, START_TABLE_HEIGHT, 0.4])
        add_reference_to_stage(table_path, START_TABLE_PATH)
        setRigidBody(self.table.prim, approximationShape='convexHull', kinematic=True)

        # box_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd"
        box_path = local_assets + '/SM_CardBoxA_02.usd'
        self.box = XFormPrim(prim_path=DEST_BOX_PATH, position=DEST_BOX_POS, scale=[1, 1, 0.4])
        add_reference_to_stage(box_path, DEST_BOX_PATH)
        setRigidBody(self.box.prim, approximationShape='convexDecomposition', kinematic=True)

        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        self.robot = UR10(prim_path=ROBOT_PATH, name='UR16e', position=ROBOT_POS, attach_gripper=True)
        # self.robot.set_joints_default_state(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))

        i = 0
        self.part = objs.DynamicCuboid(prim_path=f'{PARTS_PATH}/Part_{i}', name=f'Part_{i}',
                                       position=PARTS_SOURCE,
                                       scale=[0.1, 0.1, 0.1])
        scene.add(self.part)

        self.__camera = Camera(
            prim_path=CAMERA_PATH,
            frequency=20,
            resolution=IMG_RESOLUTION,
            # position=torch.tensor(CAMERA_POS_START),
            # orientation=torch.tensor([1, 0, 0, 0])
        )
        self.__camera.set_focal_length(2.0)

        self.__moveCamera(position=CAMERA_POS_START, target=ROBOT_POS)

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
        # self.box.initialize()
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
            img_seg_info_dict = img_seg_dict['info']['idToLabels'] # Dict: [pixel label: prim path]
            img_seg = img_seg_dict['data']  # Shape: (Width, Height)
            
            print(img_seg_info_dict)
            
            # Vectorised One-Hot-Encoding
            for label, path in img_seg_info_dict.items():
                mask = (img_seg == label) # creates a bool matrix of an element wise comparison
                print(label, path)
                if path == ROBOT_PATH:
                    one_hot_img_seg[:, :, 0] = mask
                elif path.startswith(PARTS_PATH):
                    one_hot_img_seg[:, :, 1] = mask
                elif path == DEST_BOX_PATH:
                    one_hot_img_seg[:, :, 2] = mask
        return np.concatenate([img_rgb, img_depth, one_hot_img_seg], axis=-1)



    def pre_physics_step(self, actions) -> None:
        # Rotate Joints
        joint_rots = self.robot.get_joint_positions()
        joint_rots += np.array(actions[0:6]) * self.__joint_rot_max
        self.robot.set_joint_positions(positions=joint_rots)
        # Open or close Gripper
        gripper = self.robot.gripper
        is_closed = gripper.is_closed()
        gripper_action = actions[6]
        if 0.9 < gripper_action and is_closed:
            gripper.open()
        elif gripper_action < -0.9 and not is_closed:
            gripper.close()


    
    # Calculate Rewards
    stage = 0
    def calculate_metrics(self) -> None:
        gripper = self.robot.gripper
        gripper_pos = gripper.get_world_pose()[0]
        # gripper_to_start = np.linalg.norm(START_TABLE_CENTER - gripper_pos)
        # gripper_to_dest = np.linalg.norm(DEST_BOX_POS - gripper_pos)
        
        # # Move Camera
        # curr_cam_pos = self.__camera.get_world_pose()[0]
        # closer_to_dest = gripper_to_dest < gripper_to_start
        # new_cam_pose = CAMERA_POS_DEST if closer_to_dest else CAMERA_POS_START
        # if not np.array_equal(new_cam_pose, curr_cam_pos):
        #     # cam_target = DEST_BOX_POS if closer_to_dest else START_TABLE_CENTER
        #     self.__moveCamera(new_cam_pose, ROBOT_POS)

        done = False
        reward= 0

        partPos = self.part.get_world_pose()[0]
        if self.stage != 2 and partPos[2] < 0.1:
            reward -= 100

        if self.stage == 0:
            gripper_to_part = np.linalg.norm(partPos - gripper_pos)
            reward += 1 / gripper_to_part**2
            
            if START_TABLE_HEIGHT + 0.03 < partPos[2]:
                reward += 100
                self.stage = 1

        if self.stage == 1:
            part_to_dest = np.linalg.norm(DEST_BOX_POS - partPos)
            reward += 1 / part_to_dest**2

            if part_to_dest < 0.2:
                reward += 100
                done = True
            elif partPos[2] < 0.1:
                reward -= 100

        return reward, done, {}



    def is_done(self) -> None:
        return False