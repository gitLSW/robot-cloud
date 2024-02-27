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
from omni.physx.scripts.utils import setRigidBody, setStaticCollider, setColliderSubtree, setCollider, addCollisionGroup, setPhysics, removePhysics
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
    "sim_device": "gpu",
    "enable_livestream": False,
    "task": {
        "name": 'Pack_Task',
        "physics_engine": "physx",
        "env": {
            "numEnvs": 100,
            "envSpacing": 1.5,
            "episodeLength": 100,
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
        #     'part_rot_diff': spaces.Box(low=-1, high=1, shape=(3,))
        # })
        self._num_observations = 13 + 2 * NUMBER_PARTS

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

        super().set_up_scene(scene)

        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        assets_root_path = get_assets_root_path()

        # _ = XFormPrim(prim_path=self._env_path, position=-torch.tensor([5, 4.5, 0]))
        # warehouse_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        # add_reference_to_stage(warehouse_path, self._env_path)
        
        # self.light = create_prim(
        #     '/World/Light_' + self.name,
        #     "SphereLight",
        #     position=ROBOT_POS + LIGHT_OFFSET + self._offset,
        #     attributes={
        #         "inputs:radius": 0.01,
        #         "inputs:intensity": 3e7,
        #         "inputs:color": (1.0, 1.0, 1.0)
        #     }
        # )

        # table_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        # table_path = local_assets + '/table_low.usd'
        # self.table = XFormPrim(prim_path=self._start_table_path, position=START_TABLE_POS, scale=[0.5, START_TABLE_HEIGHT, 0.4])
        # add_reference_to_stage(table_path, self._start_table_path)
        # setRigidBody(self.table.prim, approximationShape='convexHull', kinematic=True) # Kinematic True means immovable
        # # self.table = RigidPrim(rim_path=self._start_table_path, name='TABLE')
        # self._task_objects[self._start_table_path] = self.table

        env0_box_path = self.default_zero_env_path + '/box'
        box_usd_path = assets_root_path + '/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd'
        box_usd_path = local_assets + '/SM_CardBoxA_02.usd'
        add_reference_to_stage(box_usd_path, env0_box_path)
        box = XFormPrim(prim_path=env0_box_path,
                        position=DEST_BOX_POS,
                        scale=[1, 1, 0.4])
        setStaticCollider(box.prim, approximationShape='convexDecomposition')
        self._boxes_view = XFormPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/box',
                                         name='box_view',
                                         reset_xform_properties=False)
        scene.add(self._boxes_view)

        env0_part_path = self.default_zero_env_path + '/part_0'
        part_usd_path = local_assets + '/draexlmaier_part.usd'
        add_reference_to_stage(part_usd_path, env0_part_path)
        part = XFormPrim(prim_path=env0_part_path,
                         position=DEST_BOX_POS + PART_OFFSET,
                         orientation=[0, 1, 0, 0]) # [-0.70711, 0.70711, 0, 0]
        setRigidBody(part.prim, approximationShape='convexDecomposition', kinematic=True) # Kinematic True means immovable
        parts_view = XFormPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/part_0',
                                   name='part_view',
                                   reset_xform_properties=False)
        scene.add(parts_view)
        self._curr_parts = [XFormPrim(prim_path=path) for path in parts_view.prim_paths]
        
        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        env0_robot_path = self.default_zero_env_path + '/robot'
        robot = UR10(prim_path=env0_robot_path, name='UR10', position=ROBOT_POS, attach_gripper=True)
        robot.set_enabled_self_collisions(True)
        robots_view = RobotView(prim_paths_expr=f'{self.default_base_env_path}/.*/robot', name='ur10_view')
        scene.add(robots_view)
        self._robots = [UR10(prim_path=robot_path, attach_gripper=True) for robot_path in robots_view.prim_paths]
        
        # self.part_pillar = objs.FixedCuboid(
        #     name=self._pillar_path,
        #     prim_path=self._pillar_path,
        #     position=[0, 0, -100],
        #     scale=torch.tensor([1, 1, 1])
        # )
        # scene.add(self.part_pillar)

        # set_camera_view(eye=ROBOT_POS + torch.tensor([1.5, 6, 1.5]), target=ROBOT_POS, camera_prim_path="/OmniverseKit_Persp")
    
    def reset(self):
        super().reset()
        super().cleanup()
        self._placed_parts = [[] for _ in range(self._num_envs)]
        
        for env_index in range(self.num_envs):
            robot = self._robots[env_index]
            if not robot.handles_initialized:
                robot.initialize()
            self.reset_env(env_index)
    
    def reset_env(self, env_index):
        self.progress_buf[env_index] = 0
        self.reset_robot(env_index)
        self.reset_part(env_index)

    def reset_robot(self, env_index):
        robot = self._robots[env_index]
        default_pose = torch.tensor([math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0])
        robot.set_joint_positions(positions=default_pose)
        robot.gripper.open()

    def reset_part(self, env_index):
        gripper = self._robots[env_index].gripper
        part_pos = torch.tensor(gripper.get_world_pose()[0]) - torch.tensor([0, 0, 0.17], device=self._device)
        part = self._curr_parts[env_index]
        part.set_world_pose(part_pos, [0, 1, 0, 0])
        setRigidBody(part.prim, approximationShape='convexDecomposition', kinematic=False)



    # def next_part(self, env_index) -> None:
    #     gripper = self._robots[env_index].gripper
    #     gripper_pos = torch.tensor(gripper.get_world_pose()[0]) - torch.tensor([0, 0, 0.05], device=self._device)

    #     env0_part_path = self.default_zero_env_path + '/part_0'
    #     part_usd_path = local_assets + '/draexlmaier_part.usd'
    #     add_reference_to_stage(part_usd_path, env0_part_path)
    #     part = RigidPrim(prim_path=env0_part_path,
    #                     position=gripper_pos,
    #                     orientation=[0, 1, 0, 0],
    #                     mass=0.5)
    #     self.world.scene.add(part)
    #     self._placed_parts[env_index].append(self._curr_parts[env_index])
    #     self._curr_parts[env_index] = part


    # _placed_parts # [[part]] where each entry in the outer array is the placed parts for env at index
    # Returns: A 2D Array where each entry is the poses of the parts in the box
    def get_observations(self):
        def _shortest_rot_dist(quat_1, quat_2):
            part_quat = Quaternion(list(quat_1))
            ideal_quat = Quaternion(list(quat_2))
            return Quaternion.absolute_distance(part_quat, ideal_quat)
        
        boxes_pos = self._boxes_view.get_world_poses()[0] # Returns: [Array of all pos, Array of all rots]
        # obs_dicts = []
        for env_index in range(self.num_envs):
            # env_obs = { 'box_state': [] }

            robot = self._robots[env_index]
            gripper_closed = robot.gripper.is_closed()
            self.obs_buf[env_index, 0] = gripper_closed
            # env_obs['gripper_closed'] = gripper_closed
            
            robot_state = robot.get_joint_positions()
            self.obs_buf[env_index, 1:7] = robot_state

            ideal_selection = IDEAL_PACKAGING.copy()
            box_pos = boxes_pos[env_index]
            curr_part = self._curr_parts[env_index]
            eval_parts = self._placed_parts[env_index] + [curr_part]

            # box_state = []
            # ideal_pose_for_curr_part = None
            for part_index in range(NUMBER_PARTS):
                if len(eval_parts) <= part_index:
                    # The worst possible distance is 3m and 180deg
                    self.obs_buf[env_index, (13 + 2 * part_index)] = torch.scalar_tensor(3)
                    self.obs_buf[env_index, (14 + 2 * part_index)] = torch.pi
                    # env_obs['box_state'].append([3, torch.pi])
                    continue

                part_pos, part_rot = eval_parts[part_index].get_world_pose()
                part_pos -= box_pos

                ideal_part = None
                min_dist = 10000000
                # Find closest ideal part
                for pot_part in ideal_selection:
                    dist = torch.linalg.norm(torch.tensor(pot_part[0], device=self._device) - part_pos)
                    if dist < min_dist:
                        ideal_part = pot_part
                        min_dist = dist

                rot_dist = _shortest_rot_dist(part_rot, ideal_part[1])

                self.obs_buf[env_index, (13 + 2 * part_index)] = min_dist
                self.obs_buf[env_index, (14 + 2 * part_index)] = rot_dist
                # env_obs['box_state'].append([min_dist, rot_dist])

                if part_index == len(eval_parts) - 1:
                    part_pos_diff = part_pos - torch.tensor(ideal_part[0], device=self._device)
                    part_rot_euler = R.from_quat(part_rot.cpu()).as_euler('xyz', degrees=False)
                    ideal_rot_euler = R.from_quat(ideal_part[1]).as_euler('xyz', degrees=False)
                    part_rot_diff = torch.tensor(ideal_rot_euler - part_rot_euler)
                    self.obs_buf[env_index, 7:10] = part_pos_diff
                    self.obs_buf[env_index, 10:13] =  part_rot_diff
                    # env_obs['part_pos_diff'] = part_pos_diff
                    # env_obs['part_rot_diff'] = part_rot_diff
            # obs_dicts.append(env_obs)
        
        # The return is itrrelevant for Multi Threading:
        # The VecEnvMT Loop calls RLTask.post_physics_step to get all the data from one step.
        # RLTask.post_physics_step is simply returning self.obs_buf, self.rew_buf,...
        # post_physics_step calls
        # - get_observations()
        # - get_states()
        # - calculate_metrics()
        # - is_done()
        # - get_extras()
            
        # return obs_dicts
        return self.obs_buf[env_index]



    def pre_physics_step(self, actions) -> None:
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)

        for env_index in range(self.num_envs):
            if self.reset_buf[env_index]:
                self.reset_env(env_index)
                return
            
            # Rotate Joints
            robot = self._robots[env_index]
            gripper = robot.gripper

            if self.progress_buf[env_index] == 1:
                gripper.close()
                return

            joint_rots = robot.get_joint_positions()
            joint_rots += torch.tensor(actions[env_index, 0:6]) * self._max_joint_rot_speed
            robot.set_joint_positions(positions=joint_rots)

            # Open or close Gripper
            is_closed = gripper.is_closed()
            gripper_action = actions[env_index, 6]
            # if 0.9 < gripper_action and is_closed:
            #     gripper.open()
            # elif gripper_action < -0.3 and not is_closed:
            #     gripper.close()

    
    # Calculate Rewards
    def calculate_metrics(self) -> None:
        self.reset_buf = self.progress_buf >= 30

        # Terminate: Umgefallene Teile, Gefallene Teile
        # Success: 
        # part_pos, part_rot = self.part.get_world_pose()

        # any_flipped = False
        # for part in self.placed_parts:
        #     part_rot = part.get_world_pose()[1]
        #     if _is_flipped(part_rot):
        #         any_flipped = True
        #         break

        # if part_pos[2] < FALLEN_PART_THRESHOLD or self.max_steps < self.step or any_flipped:
        #     return -MAX_STEP_PUNISHMENT, True

        # box_state, _ = self.compute_box_state()
        # box_deviation = torch.sum(torch.square(box_state))

        # # placed_parts.append(self.part)

        # return -box_deviation, False
        
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


    def is_done(self):
        return False
    
def _is_flipped(q1):
    """
    Bestimmt, ob die Rotation von q0 zu q1 ein "Umfallen" darstellt,
    basierend auf einem Winkel größer als 60 Grad zwischen der ursprünglichen
    z-Achse und ihrer Rotation.

    :param q0: Ursprüngliches Quaternion.
    :param q1: Neues Quaternion.
    :return: True, wenn der Winkel größer als 60 Grad ist, sonst False.
    """
    q0 = torch.tensor([0, 1, 0, 0])
    # Initialer Vektor, parallel zur z-Achse
    v0 = torch.tensor([0, 0, 1])
    
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