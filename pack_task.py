import os
import math
import random
import numpy as np
from pxr import Gf
from gymnasium import spaces

from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.importer.urdf")
enable_extension("omni.isaac.universal_robots")
enable_extension("omni.isaac.sensor")
# from omni.importer.urdf import _urdf
from omni.isaac.sensor import Camera
from omni.isaac.universal_robots.ur10 import UR10
# from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController

from omni.isaac.core.prims import XFormPrim, RigidPrim, GeometryPrim
from omni.isaac.core.materials.physics_material import PhysicsMaterial
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


ENV_PATH = "World/Env"

ROBOT_PATH = 'World/UR10e'
ROBOT_POS = np.array([0.0, 0.0, 0.0])
# 5.45, 3, 0
START_TABLE_PATH = "World/StartTable"
START_TABLE_POS = np.array([0.36, 0.8, 0])
START_TABLE_HEIGHT = 0.6
START_TABLE_CENTER = START_TABLE_POS + np.array([0, 0, START_TABLE_HEIGHT])

DEST_BOX_PATH = "World/DestinationBox"
DEST_BOX_POS = np.array([0, -0.65, 0])

PARTS_PATH = 'World/Parts'
PARTS_SOURCE = START_TABLE_CENTER + np.array([0, 0, 0.05])
# NUM_PARTS = 5

CAMERA_PATH = 'World/Camera'
IMG_RESOLUTION = (128, 128)
CAM_TARGET_OFFSET = (2.5, 2) # Distance and Height
# CAMERA_POS_START = np.array([-2, 2, 2.5])
# CAMERA_POS_DEST = np.array([2, -2, 2.5])

MAX_STEP_PUNISHMENT = 10

class PackTask(BaseTask):
    """
    This class sets up a scene and calls a RL Policy, then evaluates the behaivior with rewards
    Args:
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        sim_s_step_freq (int): The amount of simulation steps within a SIMULATED second.
    """
    def __init__(self, name, max_steps, offset=None, sim_s_step_freq: int = 60) -> None:
        self._env_path = f"/{name}/{ENV_PATH}"
        self._robot_path = f"/{name}/{ROBOT_PATH}"
        self._start_table_path = f"/{name}/{START_TABLE_PATH}"
        self._dest_box_path = f"/{name}/{DEST_BOX_PATH}"
        self._parts_path = f"/{name}/{PARTS_PATH}"
        self._camera_path = f"/{name}/{CAMERA_PATH}"

        # self._num_observations = 1
        # self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1
        # Robot turning ange of max speed is 191deg/s
        self.__joint_rot_max = (191.0 * math.pi / 180) / sim_s_step_freq
        self.max_steps = max_steps

        self.observation_space = spaces.Dict({
            # The NN will see the Robot via a single video feed that can run from one of two camera positions
            # The NN will receive this feed in rgb, depth and image segmented to highlight objects of interest
            'image': spaces.Box(low=0, high=1, shape=(*IMG_RESOLUTION, 7)),
            # The Robot also receives the shape rotations of all 6 joints and the gripper state 
            'vector': spaces.Box(low=-1, high=1, shape=(7,)),
        })

        # The NN outputs the change in rotation for each joint as a fraction of the max rot speed per timestep (=__joint_rot_max)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=float)

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)



    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)

        local_assets = os.getcwd() + '/assets'

        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        # assets_root_path = get_assets_root_path()

        # _ = XFormPrim(prim_path=self._env_path, position=-np.array([5, 4.5, 0]))
        # warehouse_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        # add_reference_to_stage(warehouse_path, self._env_path)
        
        # scene.add_default_ground_plane()
        
        # table_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        table_path = local_assets + '/table_low.usd'
        self.table = XFormPrim(prim_path=self._start_table_path, position=START_TABLE_POS, scale=[0.5, START_TABLE_HEIGHT, 0.4])
        add_reference_to_stage(table_path, self._start_table_path)
        setRigidBody(self.table.prim, approximationShape='convexHull', kinematic=True) # Kinematic True means immovable
        # self.table = RigidPrim(rim_path=self._start_table_path, name='TABLE')
        self._task_objects[self._start_table_path] = self.table

        # box_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd"
        box_path = local_assets + '/SM_CardBoxA_02.usd'
        self.box = XFormPrim(prim_path=self._dest_box_path, position=DEST_BOX_POS, scale=[1, 1, 0.4])
        add_reference_to_stage(box_path, self._dest_box_path)
        setRigidBody(self.box.prim, approximationShape='convexDecomposition', kinematic=True) # Kinematic True means immovable
        self._task_objects[self._dest_box_path] = self.box

        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        self.robot = UR10(prim_path=self._robot_path, name='UR16e', position=ROBOT_POS, attach_gripper=True)
        self._task_objects[self._robot_path] = self.robot
        # self.robot.set_joints_default_state(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))

        i = 0
        part_usd_path = local_assets + '/dräxlmaier_part.usd'
        part_path = f'{self._parts_path}/Part_{i}'
        self.part = XFormPrim(prim_path=part_path, position=PARTS_SOURCE)
        add_reference_to_stage(part_usd_path, part_path)
        setRigidBody(self.part.prim, approximationShape='convexDecomposition', kinematic=False) # Kinematic True means immovable
        self._task_objects[self._parts_path] = self.part

        self.cam_start_pos = self.__get_cam_pos(ROBOT_POS, *CAM_TARGET_OFFSET)
        self._camera = Camera(
            prim_path=self._camera_path,
            frequency=20,
            resolution=IMG_RESOLUTION,
            # position=torch.tensor(self.cam_start_pos),
            # orientation=torch.tensor([1, 0, 0, 0])
        )
        self._camera.set_focal_length(2.0)

        self.__move_camera(position=self.cam_start_pos, target=ROBOT_POS)
        self._task_objects[self._camera_path] = self._camera

        # viewport = get_active_viewport()
        # viewport.set_active_camera(self._camera_path)

        # set_camera_view(eye=ROBOT_POS + np.array([1.5, 6, 1.5]), target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")

        self._move_task_objects_to_their_frame()
    


    def __move_camera(self, position, target):
        # USD Frame flips target and position, so they have to be flipped here
        quat = gf_quat_to_np_array(lookat_to_quatf(camera=Gf.Vec3f(*target),
                                                   target=Gf.Vec3f(*position),
                                                   up=Gf.Vec3f(0, 0, 1)))
        self._camera.set_world_pose(position=position, orientation=quat, camera_axes='usd')

    
    def __get_cam_pos(self, center, distance, height):
        angle = random.random() * 2 * math.pi
        pos = np.array([distance, 0])
        rot_matr = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        pos = np.matmul(pos, rot_matr)
        pos = np.array([*pos, height])
        return center + pos
        
        

    def reset(self):
        # super().cleanup()

        # if not self.robot.handles_initialized():
        self.robot.initialize()
        
        self._camera.initialize()
        self._camera.add_distance_to_image_plane_to_frame() # depth cam
        self._camera.add_instance_id_segmentation_to_frame() # simulated segmentation NN
        self.cam_start_pos = self.__get_cam_pos(ROBOT_POS, *CAM_TARGET_OFFSET)
        self.__move_camera(position=self.cam_start_pos, target=ROBOT_POS)

        self.step = 0
        self.stage = 0
        self.part.set_world_pose(PARTS_SOURCE)
        self.robot.set_joint_positions(positions=np.array([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))

        self._move_task_objects_to_their_frame()



    def get_observations(self):
        frame = self._camera.get_current_frame()

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
            
            # Vectorised One-Hot-Encoding
            for label, path in img_seg_info_dict.items():
                mask = (img_seg == label) # creates a bool matrix of an element wise comparison
                if path == self._self._robot_path:
                    one_hot_img_seg[:, :, 0] = mask
                elif path.startswith(self._parts_path):
                    one_hot_img_seg[:, :, 1] = mask
                elif path == self._dest_box_path:
                    one_hot_img_seg[:, :, 2] = mask
        robot_state = np.append(self.robot.get_joint_positions() / 2 * math.pi, float(self.robot.gripper.is_closed()))
        return {
            'image': np.concatenate([img_rgb, img_depth, one_hot_img_seg], axis=-1),
            'vector': robot_state
        }



    def pre_physics_step(self, actions) -> None:
        if self.step < 20:
            return
        
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
    step = 0
    def calculate_metrics(self) -> None:
        gripper = self.robot.gripper
        gripper_pos = gripper.get_world_pose()[0]
        
        # # Move Camera
        # gripper_to_start = np.linalg.norm(START_TABLE_CENTER - gripper_pos)
        # gripper_to_dest = np.linalg.norm(DEST_BOX_POS - gripper_pos)
        # curr_cam_pos = self._camera.get_world_pose()[0]
        # closer_to_dest = gripper_to_dest < gripper_to_start
        # new_cam_pose = self.cam_dest_pos if closer_to_dest else self.cam_start_pos
        # if not np.array_equal(new_cam_pose, curr_cam_pos):
        #     # cam_target = DEST_BOX_POS if closer_to_dest else START_TABLE_CENTER
        #     self.__move_camera(new_cam_pose, ROBOT_POS)

        self.step += 1
        if self.step < 20:
            return 0, False

        done = False
        reward= 0

        partPos = self.part.get_world_pose()[0]
        if self.stage == 0:
            gripper_to_part = np.linalg.norm(partPos - gripper_pos) * 100 # In cm
            reward -= max(gripper_to_part, MAX_STEP_PUNISHMENT)
            if START_TABLE_HEIGHT + 0.03 < partPos[2]: # Part was picked up
                reward += 50 * MAX_STEP_PUNISHMENT
                self.stage = 1
        elif self.stage == 1:
            part_to_dest = np.linalg.norm(DEST_BOX_POS - partPos) * 100 # In cm
            reward -= max(part_to_dest, MAX_STEP_PUNISHMENT)
            if part_to_dest < 0.2: # Part reached box
                reward += (100 + self.max_steps - self.step) * MAX_STEP_PUNISHMENT
                self.reset()
                done = True
        
        if not done and (partPos[2] < 0.1 or self.max_steps <= self.step): # Part was dropped or time ran out means end
                for _ in range(10):
                    print('END REWARD:', reward)
                reward -= (100 + self.max_steps - self.step) * MAX_STEP_PUNISHMENT
                self.reset()
                done = True

        return reward, done