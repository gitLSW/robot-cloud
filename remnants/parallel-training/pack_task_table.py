import os
import math
import random
import numpy as np
from pxr import Gf, UsdLux, Sdf
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

LEARNING_STARTS = 10

ENV_PATH = "World/Env"

ROBOT_PATH = 'World/UR10e'
ROBOT_POS = np.array([0.0, 0.0, 0.0])

LIGHT_PATH = 'World/Light'
LIGHT_OFFSET = np.array([0, 0, 2])

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

IMG_RESOLUTION = (128, 128)

CAMERA_PATH = 'World/Camera'
CAM_TARGET_OFFSET = (2.5, 2) # Distance and Height
CAM_MEAN_ANGLE = math.pi # Box=math.pi / 2, Table=3 * math.pi / 2

# CAMERA_POS_START = np.array([-2, 2, 2.5])
# CAMERA_POS_DEST = np.array([2, -2, 2.5])

MAX_STEP_PUNISHMENT = 300

IDEAL_PACKAGING = [([-0.06, -0.19984, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.14044, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.07827, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.01597, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, 0.04664, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, 0.10918, 0.0803], [0.072, 0.99, 0, 0])]

# Seed Env or DDPG will always be the same !!
class PackTask(BaseTask):
    """
    This class sets up a scene and calls a RL Policy, then evaluates the behaivior with rewards
    Args:
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        sim_s_step_freq (int): The amount of simulation steps within a SIMULATED second.
    """
    def __init__(self, name, max_steps, offset=None, sim_s_step_freq: int = 60) -> None:
        self._env_path = f"/{name}/{ENV_PATH}"
        self._light_path = {"/{name}/{LIGHT_PATH}"}
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
            # the gripper state is -1 when open and 1 when closed
            'vector': spaces.Box(low=-1, high=1, shape=(7,)),
        })

        # The NN outputs the change in rotation for each joint as a fraction of the max rot speed per timestep (=__joint_rot_max)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=float)

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)



    def set_up_scene(self, scene) -> None:
        print('SETUP TASK', self.name)

        super().set_up_scene(scene)

        local_assets = os.getcwd() + '/assets'

        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        # assets_root_path = get_assets_root_path()

        # _ = XFormPrim(prim_path=self._env_path, position=-np.array([5, 4.5, 0]))
        # warehouse_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        # add_reference_to_stage(warehouse_path, self._env_path)
        
        self.light = create_prim(
            '/World/Light_' + self.name,
            "SphereLight",
            position=ROBOT_POS + LIGHT_OFFSET + self._offset,
            attributes={
                "inputs:radius": 0.01,
                "inputs:intensity": 3e7,
                "inputs:color": (1.0, 1.0, 1.0)
            }
        )

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
        part_usd_path = local_assets + '/draexlmaier_part.usd'
        part_path = f'{self._parts_path}/Part_{i}'
        self.part = XFormPrim(prim_path=part_path, position=PARTS_SOURCE, orientation=[0, 1, 0, 0])
        add_reference_to_stage(part_usd_path, part_path)
        setRigidBody(self.part.prim, approximationShape='convexDecomposition', kinematic=False) # Kinematic True means immovable
        self._task_objects[part_path] = self.part

        cam_start_pos = self.__get_cam_pos(ROBOT_POS, *CAM_TARGET_OFFSET, mean_angle=CAM_MEAN_ANGLE)
        self._camera = Camera(
            prim_path=self._camera_path,
            frequency=20,
            resolution=IMG_RESOLUTION
        )
        self._camera.set_focal_length(2.0)

        self.__move_camera(position=cam_start_pos, target=ROBOT_POS)
        self._task_objects[self._camera_path] = self._camera

        # viewport = get_active_viewport()
        # viewport.set_active_camera(self._camera_path)

        # set_camera_view(eye=ROBOT_POS + np.array([1.5, 6, 1.5]), target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")

        self._move_task_objects_to_their_frame()
    


    def reset(self):
        # super().cleanup()

        # if not self.robot.handles_initialized():
        self.robot.initialize()
        
        self._camera.initialize()
        self._camera.add_distance_to_image_plane_to_frame() # depth cam
        self._camera.add_instance_id_segmentation_to_frame() # simulated segmentation NN
        robot_pos = ROBOT_POS + self._offset
        cam_start_pos = self.__get_cam_pos(robot_pos, *CAM_TARGET_OFFSET, mean_angle=CAM_MEAN_ANGLE)
        self.__move_camera(position=cam_start_pos, target=robot_pos)

        self.step = 0
        self.stage = 0
        # self.part.set_world_pose(PARTS_SOURCE + self._offset)
        default_pose = np.array([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0, -1])
        self.robot.gripper.open()
        self.robot.set_joint_positions(positions=default_pose[0:6])

        return {
            'image': np.zeros((*IMG_RESOLUTION, 7)),
            'vector': default_pose
        }


    def __move_camera(self, position, target):
        # USD Frame flips target and position, so they have to be flipped here
        quat = gf_quat_to_np_array(lookat_to_quatf(camera=Gf.Vec3f(*target),
                                                   target=Gf.Vec3f(*position),
                                                   up=Gf.Vec3f(0, 0, 1)))
        self._camera.set_world_pose(position=position, orientation=quat, camera_axes='usd')

    

    def __get_cam_pos(self, center, distance, height, mean_angle = None):
        angle = None
        if mean_angle:
            angle = np.random.normal(mean_angle, math.sqrt(math.pi / 16)) # Normal Distribution with mean mean_angle and sd=sqrt(10deg)
        else:
            angle = random.random() * 2 * math.pi
        pos = np.array([distance, 0])
        rot_matr = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        pos = np.matmul(pos, rot_matr)
        pos = np.array([*pos, height])
        return center + pos



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
                label = int(label)
                mask = (img_seg == label) # creates a bool matrix of an element wise comparison
                if path == self._robot_path:
                    one_hot_img_seg[:, :, 0] = mask
                elif path.startswith(self._parts_path):
                    one_hot_img_seg[:, :, 1] = mask
                elif path == self._dest_box_path:
                    one_hot_img_seg[:, :, 2] = mask

        # TODO: CHECK IF get_joint_positions ACCURATELY HANDLES ROTATIONS ABOVE 360deg AND NORMALIZE FOR MAX ROBOT ROTATION (+/-360deg)
        robot_state = np.append(self.robot.get_joint_positions() / 2 * math.pi, 2 * float(self.robot.gripper.is_closed()) - 1)
        return {
            'image': np.concatenate([img_rgb, img_depth, one_hot_img_seg], axis=-1),
            'vector': robot_state
        }



    def pre_physics_step(self, actions) -> None:
        if self.step < LEARNING_STARTS:
            return
        
        # Rotate Joints
        joint_rots = self.robot.get_joint_positions()
        joint_rots += np.array(actions[0:6]) * self.__joint_rot_max
        self.robot.set_joint_positions(positions=joint_rots)
        # Open or close Gripper
        gripper = self.robot.gripper
        is_closed = gripper.is_closed()
        gripper_action = actions[6]
        if 0.9 < gripper_action and not is_closed:
            gripper.close()
        elif gripper_action < -0.9 and is_closed:
            gripper.open()


    
    # Calculate Rewards
    stage = 0
    step = 0
    def calculate_metrics(self) -> None:
        gripper_pos = self.robot.gripper.get_world_pose()[0]

        self.step += 1
        if self.step < LEARNING_STARTS:
            return 0, False

        done = False
        reward= 0

        part_pos, part_rot = self.part.get_world_pose()
        if self.stage == 0:
            gripper_to_part = np.linalg.norm(part_pos - gripper_pos) * 100 # In cm
            reward -= gripper_to_part
            if START_TABLE_HEIGHT + 0.03 < part_pos[2]: # Part was picked up
                reward += MAX_STEP_PUNISHMENT
                self.stage = 1
        elif self.stage == 1:
            dest_box_pos = self.part.get_world_pose()[0]
            part_to_dest = np.linalg.norm(dest_box_pos - part_pos) * 100 # In cm
            print('PART TO BOX:', part_to_dest)
            if 10 < part_to_dest:
                reward -= part_to_dest
            else: # Part reached box
                # reward += (100 + self.max_steps - self.step) * MAX_STEP_PUNISHMENT
                ideal_part = self._get_closest_part(part_pos)
                pos_error = np.linalg.norm(part_pos - ideal_part[0]) * 100
                rot_error = ((part_rot - ideal_part[1])**2).mean()

                print('PART REACHED BOX:', part_to_dest)
                # print('THIS MUST BE TRUE ABOUT THE PUNISHMENT:', pos_error + rot_error, '<', MAX_STEP_PUNISHMENT) # CHeck the average punishment of stage 0 to see how much it tapers off
                reward -= pos_error + rot_error
                done = True
        

        # End successfully if all parts are in the Box
        # if self.part
        #     self.reset()
        #     done = True
        
        if not done and (part_pos[2] < 0.1 or self.max_steps <= self.step): # Part was dropped or time ran out means end
                reward -= (100 + self.max_steps - self.step) * MAX_STEP_PUNISHMENT
                # self.reset()
                done = True
        
        if done:
            print('END REWARD TASK', self.name, ':', reward)

        return reward, done
    
    def _get_closest_part(self, pos):
        pos -= self.box.get_world_pose()[0]
        closest_part = None
        min_dist = 10000000
        for part in IDEAL_PACKAGING:
            dist = np.linalg.norm(part[0] - pos)
            if dist < min_dist:
                closest_part = part
                min_dist = dist
        return closest_part