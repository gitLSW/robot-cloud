import os
import math
import random
import torch
from pxr import Gf, UsdLux, Sdf
from gymnasium import spaces

import omni.kit.commands
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
from omni.isaac.core.prims import XFormPrim, XFormPrimView, RigidPrim, RigidPrimView
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
from omni.physx.scripts.utils import setRigidBody, setStaticCollider, setColliderSubtree, setCollider, addCollisionGroup, setPhysics, removePhysics, removeRigidBody
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from omniisaacgymenvs.rl_task import RLTask
from omni.isaac.core.robots.robot_view import RobotView
from omni.isaac.cloner import GridCloner

LEARNING_STARTS = 10

FALLEN_PART_THRESHOLD = 0.2

ROBOT_PATH = 'World/UR10e'
ROBOT_POS = torch.tensor([0.0, 0.0, FALLEN_PART_THRESHOLD])

LIGHT_PATH = 'World/Light'
LIGHT_OFFSET = torch.tensor([0, 0, 2])

DEST_BOX_PATH = "World/DestinationBox"
DEST_BOX_POS = torch.tensor([0, -0.65, FALLEN_PART_THRESHOLD])

PART_PATH = 'World/Part'
PART_OFFSET = torch.tensor([0, 0, 0.4])
# NUM_PARTS = 5
PART_PILLAR_PATH = "World/Pillar"

MAX_STEP_PUNISHMENT = 300

START_TABLE_POS = torch.tensor([0.36, 0.8, 0])
START_TABLE_HEIGHT = 0.6
START_TABLE_CENTER = START_TABLE_POS + torch.tensor([0, 0, START_TABLE_HEIGHT])

IDEAL_PACKAGING = [([-0.06, -0.19984, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.14044, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.07827, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, -0.01597, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, 0.04664, 0.0803], [0.072, 0.99, 0, 0]),
                   ([-0.06, 0.10918, 0.0803], [0.072, 0.99, 0, 0])]
NUMBER_PARTS = len(IDEAL_PACKAGING)

local_assets = os.getcwd() + '/assets'

TASK_CFG = {
    "test": False,
    "device_id": 0,
    "headless": False,
    "multi_gpu": False,
    "sim_device": "cpu",
    "enable_livestream": False,
    "task": {
        "name": 'Pack_Task',
        "physics_engine": "physx",
        "env": {
            "numEnvs": 100,
            "envSpacing": 4,
            "episodeLength": 300,
            # "enableDebugVis": False,
            # "controlFrequencyInv": 4
        },
        "sim": {
            "dt": 1 / 60,
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "add_ground_plane": True,
            "use_flatcache": True,
            "enable_scene_query_support": False,
            "enable_cameras": False,
            "default_physics_material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0
            },
            "physx": {
                "worker_thread_count": 4,
                "solver_type": 1,
                "use_gpu": True,
                "solver_position_iteration_count": 4,
                "solver_velocity_iteration_count": 1,
                "contact_offset": 0.005,
                "rest_offset": 0.0,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.04,
                "friction_correlation_distance": 0.025,
                "enable_sleeping": True,
                "enable_stabilization": True,
                "max_depenetration_velocity": 1000.0,
                "gpu_max_rigid_contact_count": 524288,
                "gpu_max_rigid_patch_count": 33554432,
                "gpu_found_lost_pairs_capacity": 524288,
                "gpu_found_lost_aggregate_pairs_capacity": 262144,
                "gpu_total_aggregate_pairs_capacity": 1048576,
                "gpu_max_soft_body_contacts": 1048576,
                "gpu_max_particle_contacts": 1048576,
                "gpu_heap_capacity": 33554432,
                "gpu_temp_buffer_capacity": 16777216,
                "gpu_max_num_partitions": 8
            }
        }
    }
}


class PackTask(RLTask):
    control_frequency_inv = 1
    # kinematics_solver = None
    
    """
    This class sets up a scene and calls a RL Policy, then evaluates the behaivior with rewards
    Args:
        offset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        sim_s_step_freq (int): The amount of simulation steps within a SIMULATED second.
    """
    def __init__(self, name, sim_config, env, offset=None) -> None:
        # self.observation_space = spaces.Dict({
        #     'robot_state': spaces.Box(low=-2 * torch.pi, high=2 * torch.pi, shape=(6,)),
        #     'gripper_closed': spaces.Discrete(2),
        #     # 'forces': spaces.Box(low=-1, high=1, shape=(8, 6)), # Forces on the Joints
        #     'box_state': spaces.Box(low=-3, high=3, shape=(NUMBER_PARTS, 2)), # Pos and Rot Distance of each part currently placed in Box compared to currently gripped part
        #     'part_pos_diff': spaces.Box(low=-3, high=3, shape=(3,)),
        #     # 'part_rot_diff': spaces.Box(low=-1, high=1, shape=(3,))
        # })
        self._num_observations = 10 + 2 * NUMBER_PARTS

        # End Effector Pose
        # self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=float) # Delta Gripper Pose & gripper open / close
        self._num_actions = 7

        self.update_config(sim_config)

        # trigger __init__ of parent class
        super().__init__(name, env, offset)

    def cleanup(self):
        super().cleanup()

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.dt = self._task_cfg["sim"]["dt"]
        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        # Robot turning ange of max speed is 191deg/s
        self._max_joint_rot_speed = (191.0 * math.pi / 180) * self.dt
        
        super().update_config(sim_config)


    def set_up_scene(self, scene) -> None:
        print('SETUP TASK', self.name)
        
        self.create_env0()
        super().set_up_scene(scene) # Clones env0

        self._boxes_view = XFormPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/box',
                                         name='box_view',
                                         reset_xform_properties=False)
        scene.add(self._boxes_view)
        
        self._parts_views = []
        for i in range(NUMBER_PARTS):
            parts_view = RigidPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/part_{i}',
                                        name=f'part_{i}_view',
                                        reset_xform_properties=False)
            scene.add(parts_view)
            self._parts_views.append(parts_view)

        self._robots_view = RobotView(prim_paths_expr=f'{self.default_base_env_path}/.*/robot', name='ur10_view')
        scene.add(self._robots_view)

        # self._grippers = RigidPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/robot/ee_link', name="gripper_view")
        # scene.add(self._grippers)

        self._robots = [UR10(prim_path=robot_path, attach_gripper=True) for robot_path in self._robots_view.prim_paths]

        # self._curr_parts = [XFormPrim(prim_path=path) for path in self._parts_view.prim_paths]
        
        # # self.table = RigidPrim(rim_path=self._start_table_path, name='TABLE')
        # self._task_objects[self._start_table_path] = self.table
    
    def create_env0(self):
        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        assets_root_path = get_assets_root_path()

        env0_box_path = self.default_zero_env_path + '/box'
        # box_usd_path = assets_root_path + '/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd'
        box_usd_path = local_assets + '/SM_CardBoxA_02.usd'
        add_reference_to_stage(box_usd_path, env0_box_path)
        box = XFormPrim(prim_path=env0_box_path,
                        position=DEST_BOX_POS,
                        scale=[1, 1, 0.4])
        setStaticCollider(box.prim, approximationShape='convexDecomposition')

        for i in range(NUMBER_PARTS):
            env0_part_path = f'{self.default_zero_env_path}/part_{i}'
            part_usd_path = local_assets + '/draexlmaier_part.usd'
            add_reference_to_stage(part_usd_path, env0_part_path)
            part = RigidPrim(prim_path=env0_part_path,
                            position=START_TABLE_CENTER + torch.tensor([0, 0.1 * i - 0.2, 0.1]),
                            orientation=[0, 1, 0, 0], # [-0.70711, 0.70711, 0, 0]
                            mass=0.4)
            setRigidBody(part.prim, approximationShape='convexDecomposition', kinematic=False) # Kinematic True means immovable
        
        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        env0_robot_path = self.default_zero_env_path + '/robot'
        robot = UR10(prim_path=env0_robot_path, name='UR10', position=ROBOT_POS, attach_gripper=True)
        robot.set_enabled_self_collisions(True)

        env0_table_path = f'{self.default_zero_env_path}/table'
        table_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        # table_path = local_assets + '/table_low.usd'
        add_reference_to_stage(table_path, env0_table_path)
        table = XFormPrim(prim_path=env0_table_path,
                          position=START_TABLE_POS,
                          scale=[0.5, START_TABLE_HEIGHT, 0.4])
        setStaticCollider(table.prim, approximationShape='convexDecomposition')

    def reset(self):
        super().reset()
        super().cleanup()   
        for robot in self._robots:
            if not robot.handles_initialized:
                robot.initialize()
        indices = torch.arange(self._num_envs, dtype=torch.int64).to(self._device)
        self.reset_envs(indices)

    def reset_envs(self, env_indices):
        self.progress_buf[env_indices] = 0
        self.reset_buf[env_indices] = False
        self.reset_robots(env_indices)
        self.reset_parts(env_indices)

    def reset_robots(self, env_indices):
        default_pose = torch.tensor([math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0]).repeat(self._num_envs, 1)
        self._robots_view.set_joint_positions(default_pose, indices=env_indices)

    def reset_parts(self, env_indices):
        print('RESETTING PARTS')
        default_pos = START_TABLE_CENTER.repeat(len(env_indices), 1)
        default_rots = torch.tensor([0, 1, 0, 0]).repeat(len(env_indices), 1)
        for i in range(NUMBER_PARTS):
            parts_offsets = torch.tensor([0, 0.1 * i - 0.2, 0.1]).repeat(len(env_indices), 1)
            self._parts_views[i].set_world_poses(positions=default_pos + parts_offsets + self._env_pos[i].cpu(),
                                                 orientations=default_rots,
                                                 indices=env_indices)

    # _placed_parts # [[part]] where each entry in the outer array is the placed parts for env at index
    # Returns: A 2D Array where each entry is the poses of the parts in the box
    def get_observations(self):
        def _shortest_rot_dist(quat_1, quat_2):
            part_quat = Quaternion(list(quat_1))
            ideal_quat = Quaternion(list(quat_2))
            return Quaternion.absolute_distance(part_quat, ideal_quat)
        
        boxes_pos = self._boxes_view.get_world_poses()[0] # Returns: [Array of all pos, Array of all rots]

        robots_states = self._robots_view.get_joint_positions()
        self.obs_buf[:, 1:7] = robots_states
        
        parts_pos = []
        parts_rots = []
        for parts_view in self._parts_views:
            curr_parts_pos, curr_parts_rots = parts_view.get_world_poses()
            curr_parts_pos -= boxes_pos
            parts_pos.append(curr_parts_pos)
            parts_rots.append(curr_parts_rots)

        parts_pos = torch.stack(parts_pos).transpose(0, 1) # Stacks and transposes the array
        parts_rots = torch.stack(parts_rots).transpose(0, 1)
        
        for env_index in range(self._num_envs):
            gripper_closed = self._robots[env_index].gripper.is_closed()
            self.obs_buf[env_index, 0] = gripper_closed

            ideal_selection = IDEAL_PACKAGING.copy()

            gripper_pos = self._robots[env_index].gripper.get_world_pose()[0]
            gripper_pos -= boxes_pos[env_index]

            gripper_to_closest_part_dist = 100000
            gripper_to_closest_part_dir = None

            gripper_to_ideal_part_dist = 100000
            gripper_to_ideal_part_dir = None
            for part_index in range(NUMBER_PARTS):
                part_pos = parts_pos[env_index][part_index]
                part_rot = parts_rots[env_index][part_index]

                part_to_box_dist = torch.linalg.norm(part_pos)
                if 0.3 < part_to_box_dist: # Only parts that are not packed can be potential gripped parts
                    gripper_to_part = part_pos - gripper_pos
                    gripper_part_dist = torch.linalg.norm(gripper_to_part)
                    if gripper_part_dist < gripper_to_closest_part_dist:
                        gripper_to_closest_part_dist = gripper_part_dist
                        gripper_to_closest_part_dir = gripper_to_part

                ideal_part = None
                min_dist = 10000000
                # Find closest ideal part
                for ideal_part in ideal_selection:
                    ideal_part_pos = torch.tensor(ideal_part[0]).to(self._device)
                    dist = torch.linalg.norm(ideal_part_pos - part_pos)
                    if dist < min_dist:
                        ideal_part = ideal_part
                        min_dist = dist
                    if part_index == 0: # Only one check is needed
                        gripper_to_ideal_part = ideal_part_pos - gripper_pos
                        gripper_ideal_dist = torch.linalg.norm(gripper_to_ideal_part)
                        if gripper_ideal_dist < gripper_to_ideal_part_dist:
                            gripper_to_ideal_part_dist = gripper_ideal_dist
                            gripper_to_ideal_part_dir = gripper_to_ideal_part


                rot_dist = _shortest_rot_dist(part_rot, ideal_part[1])
                
                # Clip obs
                min_dist = min(min_dist, 3)
                rot_dist = min(rot_dist, torch.pi)

                # Record obs
                self.obs_buf[env_index, (10 + part_index)] = min_dist
                self.obs_buf[env_index, (10 + NUMBER_PARTS + part_index)] = rot_dist

            # Point to target
            self.obs_buf[env_index, 7:10] = gripper_to_ideal_part_dir if gripper_to_closest_part_dist < 0.05 else gripper_to_closest_part_dir


    def pre_physics_step(self, actions) -> None:
        reset_env_indices = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_indices) > 0:
            self.reset_envs(reset_env_indices)

        # Rotate Joints
        joint_rots = self._robots_view.get_joint_positions()
        joint_rots += torch.tensor(actions[:, 0:6]) * self._max_joint_rot_speed
        self._robots_view.set_joint_positions(positions=joint_rots)

        # Open or close Gripper
        for env_index in range(self._num_envs):
            gripper = self._robots[env_index].gripper
            is_closed = gripper.is_closed()
            gripper_action = actions[env_index, 6]
            if 0.9 < gripper_action and is_closed:
                gripper.open()
            elif gripper_action < -0.3 and not is_closed:
                gripper.close()
    
    # Calculate Rewards
    def calculate_metrics(self) -> None:
        targets_dirs = self.obs_buf[:, 7:10]
        targets_dists = torch.linalg.norm(targets_dirs, dim=1)

        # part_rot_diffs = self.obs_buf[:, 10:13]
        ideal_pos_dists = self.obs_buf[:, 10:(10 + NUMBER_PARTS)]
        ideal_rot_dists = self.obs_buf[:, (10 + NUMBER_PARTS):(10 + 2 * NUMBER_PARTS)]

        box_error_sum = ideal_pos_dists.square().sum(dim=1) + ideal_rot_dists.abs().sum(dim=1)
        self.rew_buf = -targets_dists.square() - box_error_sum

    def is_done(self):
        # any_flipped = False
        self.reset_buf.fill_(0)
        for parts_view in self._parts_views:
            parts_pos = parts_view.get_world_poses()[0]

            # Check if part has fallen
            # self.reset_buf += (parts_pos[:, 2] < FALLEN_PART_THRESHOLD)

            # if _is_flipped(part_rot):
            #     any_flipped = True
            #     break

        self.reset_buf += (self._max_episode_length - 1 <= self.progress_buf)
        self.reset_buf = self.reset_buf >= 1
    
# def _is_flipped(q1):
#     """
#     Bestimmt, ob die Rotation von q0 zu q1 ein "Umfallen" darstellt,
#     basierend auf einem Winkel größer als 60 Grad zwischen der ursprünglichen
#     z-Achse und ihrer Rotation.

#     :param q0: Ursprüngliches Quaternion.
#     :param q1: Neues Quaternion.
#     :return: True, wenn der Winkel größer als 60 Grad ist, sonst False.
#     """
#     q0 = torch.tensor([0, 1, 0, 0])
#     # Initialer Vektor, parallel zur z-Achse
#     v0 = torch.tensor([0, 0, 1])
    
#     # Konvertiere Quaternions in Rotation-Objekte
#     rotation0 = R.from_quat(q0)
#     rotation1 = R.from_quat(q1)
    
#     # Berechne die relative Rotation von q0 zu q1
#     q_rel = rotation1 * rotation0.inv()
    
#     # Berechne den rotierten Vektor v1
#     v1 = q_rel.apply(v0)
    
#     # Berechne den Winkel zwischen v0 und v1
#     cos_theta = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
#     angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    
#     # Prüfe, ob der Winkel größer als 60 Grad ist
#     return angle > 60