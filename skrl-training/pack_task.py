import os
import math
import random
import torch
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
from pyquaternion import Quaternion

from omni.isaac.gym.tasks.rl_task import RLTaskInterface
from omni.isaac.cloner import GridCloner

LEARNING_STARTS = 10

ENV_PATH = "World/Env"

FALLEN_PART_THRESHOLD = 0.2

ROBOT_PATH = 'World/UR10e'
ROBOT_POS = np.array([0.0, 0.0, FALLEN_PART_THRESHOLD])

LIGHT_PATH = 'World/Light'
LIGHT_OFFSET = np.array([0, 0, 2])

DEST_BOX_PATH = "World/DestinationBox"
DEST_BOX_POS = np.array([0, -0.65, FALLEN_PART_THRESHOLD])

PART_PATH = 'World/Part'
PART_SOURCE = DEST_BOX_POS + np.array([0, 0, 0.4])
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


class PackTask(RLTaskInterface):
    kinematics_solver = None
    
    """
    This class sets up a scene and calls a RL Policy, then evaluates the behaivior with rewards
    Args:
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        sim_s_step_freq (int): The amount of simulation steps within a SIMULATED second.
    """
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._env_path = f"/{name}/{ENV_PATH}"
        self._light_path = {"/{name}/{LIGHT_PATH}"}
        self._robot_path = f"/{name}/{ROBOT_PATH}"
        self._dest_box_path = f"/{name}/{DEST_BOX_PATH}"
        self._part_path = f"/{name}/{PART_PATH}"
        self._pillar_path = f"/{name}/{PART_PILLAR_PATH}"

        # self._num_observations = 1
        # self._num_actions = 1

        self.observation_space = spaces.Dict({
            'gripper_closed': spaces.Discrete(2),
            # 'forces': spaces.Box(low=-1, high=1, shape=(8, 6)), # Forces on the Joints
            'box_state': spaces.Box(low=-3, high=3, shape=(NUMBER_PARTS, 2)), # Pos and Rot Distance of each part currently placed in Box compared to currently gripped part
            'part_state': spaces.Box(low=-3, high=3, shape=(6,))
        })

        # End Effector Pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=float) # Delta Gripper Pose & gripper open / close

        self.update_config(sim_config)

        # trigger __init__ of parent class
        super.__init__(self, name=name, env=env, offset=offset)
        

    def update_config(self, sim_config):
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_observations = 60
        self._num_actions = 8
        self._ant_positions = torch.tensor([0, 0, 0.5])

        self._device = self._cfg["sim_device"]
        
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]

        # Robot turning ange of max speed is 191deg/s
        self.__joint_rot_max = (191.0 * math.pi / 180) / 60 # The sim runs on 60hz by default
        
        RLTaskInterface.update_config(self)


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
            position=[0, 0, -100],
            scale=np.array([1, 1, 1])
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
        default_pose = np.array([math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0])
        self.robot.set_joint_positions(positions=default_pose)

        if not self.kinematics_solver:
            self.kinematics_solver = KinematicsSolver(robot_articulation=self.robot, attach_gripper=True)

        self.step = 0

        gripper_pos = np.array(self.robot.gripper.get_world_pose()[0]) - np.array([0, 0, 0.25])
        self.part_pillar.set_world_pose([gripper_pos[0], gripper_pos[1], gripper_pos[2] / 2])
        self.part_pillar.set_local_scale([1, 1, gripper_pos[2]])
        self.part.set_world_pose(gripper_pos, [0, 1, 0, 0])

        # gripper_pos, gripper_rot = self.robot.gripper.get_world_pose()
        # gripper_pos -= self.robot.get_world_pose()[0]
        # self.robot.gripper.open()
        #  np.concatenate((gripper_pos, gripper_rot, [-1]), axis=0)


        box_state, current_ideal_pose = self.compute_box_state()

        part_pos, part_rot = self.part.get_world_pose()
        part_pos -= self.box.get_world_pose()[0]
        part_rot_euler = R.from_quat(part_rot).as_euler('xyz',degrees=False)
        ideal_rot_euler = R.from_quat(current_ideal_pose[1]).as_euler('xyz',degrees=False)
        part_pos_diff = current_ideal_pose[0] - part_pos
        part_rot_diff = ideal_rot_euler - part_rot_euler

        return {
            'gripper_closed': False,
            'box_state': box_state,
            'part_state': np.concatenate((part_pos_diff, part_rot_diff), axis=0)
        }


    placed_parts = []
    def get_observations(self):
        # gripper = self.robot.gripper
        # gripper_pos, gripper_rot = gripper.get_world_pose()
        # gripper_pos -= self.robot.get_world_pose()[0]

        # gripper_pose_kin = self.kinematics_solver.compute_end_effector_pose()
        # gripper_closed = 2 * float(gripper.is_closed()) - 1
        # # 'gripper_state': np.concatenate((gripper_pos, gripper_rot, [gripper_closed]), axis=0),

        # # TODO: Was sollen wir verwenden ??!!
        # print('COMPARE:')
        # print(gripper_pose_kin)
        # print((gripper_pos, gripper_rot))

        # forces = self.robot.get_measured_joint_forces()

        box_state, current_ideal_pose = self.compute_box_state()

        part_pos, part_rot = self.part.get_world_pose()
        part_pos -= self.box.get_world_pose()[0]
        part_rot_euler = R.from_quat(part_rot).as_euler('xyz',degrees=False)
        ideal_rot_euler = R.from_quat(current_ideal_pose[1]).as_euler('xyz',degrees=False)
        part_pos_diff = current_ideal_pose[0] - part_pos
        part_rot_diff = ideal_rot_euler - part_rot_euler

        return {
            'gripper_closed': self.robot.gripper.is_closed(),
            'box_state': box_state,
            # 'forces': forces,
            'part_state':  np.concatenate((part_pos_diff, part_rot_diff), axis=0)
        }


    # Returns: A 2D Array where each entry is the poses of the parts in the box
    def compute_box_state(self):
        box_state = []
        ideal_selection = IDEAL_PACKAGING.copy()
        parts = self.placed_parts + [self.part]
        current_ideal_pose = None

        for i in range(NUMBER_PARTS):
            if len(parts) <= i:
                box_state.append([3, math.pi])
                continue
            part = parts[i]
            part_pos, part_rot= part.get_world_pose()
            part_pos -= self.box.get_world_pose()[0]

            ideal_part = None
            min_dist = 10000000
            # Find closest ideal part
            for sel_part in ideal_selection:
                dist = np.linalg.norm(sel_part[0] - part_pos)
                if dist < min_dist:
                    ideal_part = sel_part
                    min_dist = dist
                    if i == len(parts) - 1:
                        current_ideal_pose = ideal_part

            ideal_selection.remove(ideal_part)
            rot_dist = _shortest_rot_dist(part_rot, ideal_part[1])
            box_state.append([min_dist, rot_dist])

        return box_state, current_ideal_pose


    def pre_physics_step(self, actions) -> None:
        gripper = self.robot.gripper
        
        if self.step == LEARNING_STARTS - 1:
            gripper.close()
            return
        elif self.step == LEARNING_STARTS:
            self.part_pillar.set_world_pose([0, 0, -100])
            return
        elif self.step < LEARNING_STARTS:
            return
        
        # Rotate Joints
        gripper_pos = actions[0:3]
        gripper_rot_euler = actions[3:6]
        gripper_action = actions[6]

        gripper_rot = R.from_euler('xyz', gripper_rot_euler, degrees=False).as_quat()
        movement, success = self.kinematics_solver.compute_inverse_kinematics(gripper_pos, gripper_rot)
        # print('success', success)
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
        self.step += 1
        # Terminate: Umgefallene Teile, Gefallene Teile
        # Success: 
        part_pos, part_rot = self.part.get_world_pose()

        any_flipped = False
        for part in self.placed_parts:
            part_rot = part.get_world_pose()[1]
            if _is_flipped(part_rot):
                any_flipped = True
                break

        if part_pos[2] < FALLEN_PART_THRESHOLD or self.max_steps < self.step or any_flipped:
            return -MAX_STEP_PUNISHMENT, True

        box_state, _ = self.compute_box_state()
        box_deviation = np.sum(np.square(box_state))

        # placed_parts.append(self.part)

        return -box_deviation, False
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
    
def _is_flipped(q1):
    """
    Bestimmt, ob die Rotation von q0 zu q1 ein "Umfallen" darstellt,
    basierend auf einem Winkel größer als 60 Grad zwischen der ursprünglichen
    z-Achse und ihrer Rotation.

    :param q0: Ursprüngliches Quaternion.
    :param q1: Neues Quaternion.
    :return: True, wenn der Winkel größer als 60 Grad ist, sonst False.
    """
    q0 = np.array([0, 1, 0, 0])
    # Initialer Vektor, parallel zur z-Achse
    v0 = np.array([0, 0, 1])
    
    # Konvertiere Quaternions in Rotation-Objekte
    rotation0 = R.from_quat(q0)
    rotation1 = R.from_quat(q1)
    
    # Berechne die relative Rotation von q0 zu q1
    q_rel = rotation1 * rotation0.inv()
    
    # Berechne den rotierten Vektor v1
    v1 = q_rel.apply(v0)
    
    # Berechne den Winkel zwischen v0 und v1
    cos_theta = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    
    # Prüfe, ob der Winkel größer als 60 Grad ist
    return angle > 60

def _shortest_rot_dist(quat_1, quat_2):
    part_quat = Quaternion(quat_1)
    ideal_quat = Quaternion(quat_2)
    return Quaternion.absolute_distance(part_quat, ideal_quat)