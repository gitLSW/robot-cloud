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
from omni.isaac.universal_robots import KinematicsSolver
# from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController

import omni.isaac.core.utils.prims as prims_utils
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
from scipy.spatial.transform import Rotation as R

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

DEST_BOX_PATH = "World/DestinationBox"
DEST_BOX_POS = np.array([0, -0.65, 0])

PART_PATH = 'World/Part'
PART_HEIGHT = 0.6
PART_SOURCE = DEST_BOX_POS + np.array([0, 0, PART_HEIGHT])
# NUM_PARTS = 5
PART_PILLAR_PATH = "World/Pillar"

MAX_STEP_PUNISHMENT = 300

IDEAL_PACKAGING = [([-0.06, -0.19984, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.14044, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.07827, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.01597, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, 0.04664, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, 0.10918, 0.0803], [0.072, 0.99, 0, 0])]
NUMBER_PARTS = len(IDEAL_PACKAGING)

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
        self._dest_box_path = f"/{name}/{DEST_BOX_PATH}"
        self._part_path = f"/{name}/{PART_PATH}"
        self._pillar_path = f"/{name}/{PART_PILLAR_PATH}"

        # self._num_observations = 1
        # self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1
        # Robot turning ange of max speed is 191deg/s
        self.__joint_rot_max = (191.0 * math.pi / 180) / sim_s_step_freq
        self.max_steps = max_steps

        self.observation_space = spaces.Dict({
            'gripper_state': spaces.Box(low=-3, high=-3, shape=(8,)),
            # 'forces': spaces.Box(low=-1, high=1, shape=(8, 6)), # Forces on the Joints
            'box_state': spaces.Box(low=-3, high=-3, shape=(2, NUMBER_PARTS)), # Pos and Rot Distance of each part currently placed in Box compared to currently gripped part
            'part_state': spaces.Box(low=-3, high=-3, shape=(7,))
        })

        # End Effector Pose 
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,)), # Delta Gripper Pose & gripper open / close

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

        # box_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd"
        box_path = local_assets + '/SM_CardBoxA_02.usd'
        self.box = XFormPrim(prim_path=self._dest_box_path, position=DEST_BOX_POS, scale=[1, 1, 0.4])
        add_reference_to_stage(box_path, self._dest_box_path)
        setRigidBody(self.box.prim, approximationShape='convexDecomposition', kinematic=True) # Kinematic True means immovable
        self._task_objects[self._dest_box_path] = self.box

        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        self.robot = UR10(prim_path=self._robot_path, name='UR10', position=ROBOT_POS, attach_gripper=True)
        self._task_objects[self._robot_path] = self.robot
        # self.robot.set_joints_default_state(positions=torch.tensor([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]))

        self.part_pillar = objs.FixedCuboid(
            name=self._pillar_path,
            prim_path=self._pillar_path,
            position=DEST_BOX_POS,
            scale=np.array([0.2, 0.2, PART_HEIGHT])
        )
        scene.add(self.part_pillar)
        self._task_objects[self._pillar_path] = self.part_pillar

        part_usd_path = local_assets + '/draexlmaier_part.usd'
        self.part = XFormPrim(prim_path=self._part_path, position=PART_SOURCE, orientation=[0, 1, 0, 0])
        add_reference_to_stage(part_usd_path, self._part_path)
        setRigidBody(self.part.prim, approximationShape='convexDecomposition', kinematic=False) # Kinematic True means immovable
        self._task_objects[self._part_path] = self.part

        # set_camera_view(eye=ROBOT_POS + np.array([1.5, 6, 1.5]), target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")

        self._move_task_objects_to_their_frame()
    


    def reset(self):
        # super().cleanup()

        # if not self.robot.handles_initialized():
        self.robot.initialize()
        self.kinematics_solver = KinematicsSolver(robot_articulation=self.robot, attach_gripper=True)

        self.step = 0
        self.part.set_world_pose(PART_SOURCE + self._offset)
        default_pose = np.array([math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0])
        self.robot.set_joint_positions(positions=default_pose)

        gripper_pos, gripper_rot = self.robot.gripper.get_world_pose()
        self.robot.gripper.open()

        return {
            'gripper_state': np.array(gripper_pos + gripper_rot + [-1]),
            'box_state': default_pose,
            'part_state': self.part.get_world_pose()
        }


    placed_parts = []
    def get_observations(self):
        gripper = self.robot.gripper
        gripper_pose_kin = self.kinematics_solver.compute_end_effector_pose()
        gripper_pos, gripper_rot = gripper.get_world_pose()
        gripper_pos -= self.robot.get_world_pose()
        gripper_closed = 2 * float(gripper.is_closed()) - 1

        print('COMPARE', gripper_pose_kin, gripper.get_world_pose())

        # A 2D Array where each entry is the poses of the parts in the box
        part_pos , part_rot= self.part.get_world_pose()
        part_pos -= self.box.get_world_pose()[0]
        box_state = []
        selection = IDEAL_PACKAGING
        for part in self.placed_parts:
            ideal_rot = None
            ideal_part_index = None
            min_dist = 10000000
            for index, part in enumerate(selection):
                dist = np.linalg.norm(part[0] - part_pos)
                if dist < min_dist:
                    ideal_rot = part[1]
                    ideal_part_index = index
                    min_dist = dist
            selection.remove(ideal_part_index)
            rot_dist = self._shortest_rot_dist(part_rot, ideal_rot)
            box_state.append([min_dist, rot_dist])

        # forces = self.robot.get_measured_joint_forces()
        return {
            'gripper_state': np.array(gripper_pos + gripper_rot + [gripper_closed]),
            'bos_state': box_state,
            # 'forces': forces,
            'part_state': np.array(part_pos + part_rot)
        }



    def pre_physics_step(self, actions) -> None:
        gripper = self.robot.gripper
        
        if self.step == LEARNING_STARTS -1:
            gripper.close()
            return
        elif self.step == LEARNING_STARTS:
            prims_utils.delete_prim(self._pillar_path)
            return
        elif self.step < LEARNING_STARTS:
            return
        
        # Rotate Joints
        gripper_pos = actions[0:2]
        gripper_rot = actions[3:6]
        gripper_action = actions[7]

        movement, success = self.kinematics_solver.compute_inverse_kinematics(gripper_pos, gripper_rot)
        if success:
            self.robot.apply_action(movement)

        is_closed = gripper.is_closed()
        if 0.9 < gripper_action and not is_closed:
            gripper.close()
        elif gripper_action < -0.9 and is_closed:
            gripper.open()


    
    # Calculate Rewards
    step = 0
    def calculate_metrics(self) -> None:
        return 0, False
        # gripper_pos = self.robot.gripper.get_world_pose()[0]

        # self.step += 1
        # if self.step < LEARNING_STARTS:
        #     return 0, False

        # done = False
        # reward= 0

        # part_pos, part_rot = self.part.get_world_pose()
        # dest_box_pos = self.part.get_world_pose()[0]
        # part_to_dest = np.linalg.norm(dest_box_pos - part_pos) * 100 # In cm

        # print('PART TO BOX:', part_to_dest)
        # if 10 < part_to_dest:
        #     reward -= part_to_dest
        # else: # Part reached box
        #     # reward += (100 + self.max_steps - self.step) * MAX_STEP_PUNISHMENT
        #     ideal_part = _get_closest_part(part_pos)
        #     pos_error = np.linalg.norm(part_pos - ideal_part[0]) * 100
        #     rot_error = ((part_rot - ideal_part[1])**2).mean()

        #     print('PART REACHED BOX:', part_to_dest)
        #     # print('THIS MUST BE TRUE ABOUT THE PUNISHMENT:', pos_error + rot_error, '<', MAX_STEP_PUNISHMENT) # CHeck the average punishment of stage 0 to see how much it tapers off
        #     reward -= pos_error + rot_error

        # # if not done and (part_pos[2] < 0.1 or self.max_steps <= self.step): # Part was dropped or time ran out means end
        # #         reward -= (100 + self.max_steps - self.step) * MAX_STEP_PUNISHMENT
        # #         done = True
        
        # if done:
        #     print('END REWARD TASK', self.name, ':', reward)

        # return reward, done
    


def _shortest_rot_dist(quat_1, quat_2):
        quat_1 = R.from_quat(quat_1)
        quat_2 = R.from_quat(quat_2)
        if (np.dot(quat_1, quat_2) < 0):
            return quat_1 * R.inv(-quat_2)
        else:
            return quat_1 * R.inv(quat_2)