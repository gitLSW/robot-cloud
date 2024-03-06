import os
import math
import torch
from gymnasium import spaces

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.universal_robots")
# enable_extension("omni.isaac.sensor")

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import delete_prim #, create_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts.utils import setRigidBody, setStaticCollider #, setColliderSubtree, setCollider, addCollisionGroup, setPhysics, removePhysics, removeRigidBody

from omni.isaac.universal_robots.ur10 import UR10
from omni.isaac.core.prims import XFormPrim, XFormPrimView, RigidPrim, RigidPrimView
from omni.isaac.core.robots.robot_view import RobotView
# from omni.isaac.core.materials.physics_material import PhysicsMaterial

from omniisaacgymenvs.rl_task import RLTask
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


FALLEN_PART_THRESHOLD = 0.2

ROBOT_POS = torch.tensor([0.0, 0.0, FALLEN_PART_THRESHOLD])

LIGHT_OFFSET = torch.tensor([0, 0, 2])

DEST_BOX_POS = torch.tensor([0, -0.65, FALLEN_PART_THRESHOLD])

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
        # "physics_engine": "physx",
        "env": {
            "numEnvs": 625,
            "envSpacing": 4,
            "episodeLength": 30, # The episode length is the max time for one part to be packed, not the whole box
            # "enableDebugVis": False,
            # "controlFrequencyInv": 4
        },
        "sim": {
            "dt": 1.0 / 60.0,
            "gravity": [0.0, 0.0, -9.81],
            "substeps": 1,
            "use_gpu_pipeline": False, # Must be off for gripper to work
            "add_ground_plane": True,
            "add_distant_light": True,
            "use_fabric": True,
            "enable_scene_query_support": True, # Must be on for gripper to work
            "enable_cameras": False,
            "disable_contact_processing": False, # Must be off for gripper to work
            "use_flatcache": True,
            "default_physics_material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0
            },
            "physx": {
                ### Per-scene settings
                "use_gpu": True,
                "worker_thread_count": 4,
                "solver_type": 1,  # 0: PGS, 1:TGS
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.04,  # A threshold of contact separation distance used to decide if a contact
                # point will experience friction forces.
                "friction_correlation_distance": 0.025,  # Contact points can be merged into a single friction anchor if the
                # distance between the contacts is smaller than correlation distance.
                # disabling these can be useful for debugging
                "enable_sleeping": True,
                "enable_stabilization": True,
                # GPU buffers
                "gpu_max_rigid_contact_count": 512 * 1024,
                "gpu_max_rigid_patch_count": 80 * 1024,
                "gpu_found_lost_pairs_capacity": 1024,
                "gpu_found_lost_aggregate_pairs_capacity": 1024,
                "gpu_total_aggregate_pairs_capacity": 1024,
                "gpu_max_soft_body_contacts": 1024 * 1024,
                "gpu_max_particle_contacts": 1024 * 1024,
                "gpu_heap_capacity": 64 * 1024 * 1024,
                "gpu_temp_buffer_capacity": 16 * 1024 * 1024,
                "gpu_max_num_partitions": 8,
                "gpu_collision_stack_size": 64 * 1024 * 1024,
                ### Per-actor settings ( can override in actor_options )
                "solver_position_iteration_count": 4,
                "solver_velocity_iteration_count": 1,
                "sleep_threshold": 0.0,  # Mass-normalized kinetic energy threshold below which an actor may go to sleep.
                # Allowed range [0, max_float).
                "stabilization_threshold": 0.0,  # Mass-normalized kinetic energy threshold below which an actor may
                # participate in stabilization. Allowed range [0, max_float).
                ### Per-body settings ( can override in actor_options )
                "enable_gyroscopic_forces": False,
                "density": 1000.0,  # density to be used for bodies that do not specify mass or density
                "max_depenetration_velocity": 100.0,
                ### Per-shape settings ( can override in actor_options )
                "contact_offset": 0.02,
                "rest_offset": 0.001,
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
        self._num_observations = 10 + 2 * NUMBER_PARTS
        # self.observation_space = spaces.Dict({ 'obs': spaces.Box(low=-math.pi, high=math.pi, shape=(self._num_observations,), dtype=float) })
        
        self._num_actions = 7 # gripper open / close & Delta 6 joint rots
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self._num_actions,), dtype=float)

        self.update_config(sim_config)
        super().__init__(name, env, offset)

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
        self._max_joint_rot_speed = torch.scalar_tensor((191.0 * torch.pi / 180) * self.dt).to(self._device)
        
        super().update_config(sim_config)


    def set_up_scene(self, scene) -> None:
        print('SETUP TASK', self.name)
        
        self.create_env0()
        super().set_up_scene(scene) # Clones env0

        self._boxes_view = XFormPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/box',
                                         name='box_view',
                                         reset_xform_properties=False)
        scene.add(self._boxes_view)
        
        # parts_view = RigidPrimView(prim_paths_expr='{self.default_base_env_path}/.*/part_0',
        #                             name='part_0_view',
        #                             reset_xform_properties=False)
        # scene.add(parts_view)

        self._robots_view = RobotView(prim_paths_expr=f'{self.default_base_env_path}/.*/robot', name='ur10_view')
        scene.add(self._robots_view)

        self._grippers = RigidPrimView(prim_paths_expr=f'{self.default_base_env_path}/.*/robot/ee_link', name="gripper_view")
        scene.add(self._grippers)

        # self._curr_parts = [RigidPrim(prim_path=path) for path in parts_view.prim_paths]
        self._robots = [UR10(prim_path=robot_path, attach_gripper=True) for robot_path in self._robots_view.prim_paths]
    
    def create_env0(self):
        # This is the URL from which the Assets are downloaded
        # Make sure you started and connected to your localhost Nucleus Server via Omniverse !!!
        assets_root_path = get_assets_root_path()

        env0_box_path = self.default_zero_env_path + '/box'
        box_usd_path = assets_root_path + '/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd'
        box_usd_path = local_assets + '/SM_CardBoxA_02.usd'
        add_reference_to_stage(box_usd_path, env0_box_path)
        box = XFormPrim(prim_path=env0_box_path,
                        position=DEST_BOX_POS,
                        scale=[1, 1, 0.4])
        setStaticCollider(box.prim, approximationShape='convexDecomposition')

        # The UR10e has 6 joints, each with a maximum:
        # turning angle of -360 deg to +360 deg
        # turning ange of max speed is 191deg/s
        env0_robot_path = self.default_zero_env_path + '/robot'
        robot = UR10(prim_path=env0_robot_path, name='UR10', position=ROBOT_POS, attach_gripper=True)
        robot.set_enabled_self_collisions(True)

    def cleanup(self):
        self._curr_parts = [None for _ in range(self._num_envs)]
        self._placed_parts = [[] for _ in range(self._num_envs)]
        super().cleanup()

    def reset(self):
        super().reset()
        self.cleanup()
        for env_index in range(self._num_envs):
            robot = self._robots[env_index]
            if not robot.handles_initialized:
                robot.initialize()
            self.reset_env(env_index)

    def reset_env(self, env_index):
        self.progress_buf[env_index] = 0
        self.reset_buf[env_index] = False
        
        curr_part = self._curr_parts[env_index]
        if curr_part:
            delete_prim(curr_part.prim_path)
            self._curr_parts[env_index] = None
            
        for part in self._placed_parts[env_index]:
            delete_prim(part.prim_path)
        self._placed_parts[env_index] = []

        self.reset_robot(env_index)

    def reset_robot(self, env_index):
        robot = self._robots[env_index]
        default_pose = torch.tensor([math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2, 0])
        robot.set_joint_positions(positions=default_pose)
        robot.gripper.open()

    def add_part(self, env_index) -> None:
        gripper = self._robots[env_index].gripper
        part_pos = torch.tensor(gripper.get_world_pose()[0]) - torch.tensor([0, 0, 0.05], device=self._device)

        part_index = len(self._placed_parts[env_index])
        part_path = f'{self.default_base_env_path}/env_{env_index}/parts/part_{part_index}'
        print(part_path, part_index)
        part_usd_path = local_assets + '/draexlmaier_part.usd'
        add_reference_to_stage(part_usd_path, part_path)
        part = RigidPrim(prim_path=part_path,
                        name=f'env_{env_index}_part_{part_index}',
                        position=part_pos,
                        orientation=[0, 1, 0, 0],
                        mass=0.4) # [-0.70711, 0.70711, 0, 0]
        setRigidBody(part.prim, approximationShape='convexDecomposition', kinematic=True) # Kinematic True means immovable
        return part


    # _placed_parts # [[part]] where each entry in the outer array is the placed parts for env at index
    # Returns: A 2D Array where each entry is the poses of the parts in the box
    def get_observations(self):
        def _shortest_rot_dist(quat_1, quat_2):
            part_quat = Quaternion(list(quat_1))
            ideal_quat = Quaternion(list(quat_2))
            return Quaternion.absolute_distance(part_quat, ideal_quat)
        
        boxes_pos = self._boxes_view.get_world_poses()[0] # Returns: [Array of all pos, Array of all rots]
        # obs_dicts = []
        for env_index in range(self._num_envs):
            # env_obs = { 'box_state': [] }

            robot = self._robots[env_index]
            gripper_closed = robot.gripper.is_closed()
            self.obs_buf[env_index, 0] = gripper_closed
            # env_obs['gripper_closed'] = gripper_closed
            
            robot_state = robot.get_joint_positions()
            self.obs_buf[env_index, 1:7] = robot_state

            ideal_selection = IDEAL_PACKAGING.copy()
            box_pos = boxes_pos[env_index]
            eval_parts = self._placed_parts[env_index]
            curr_part = self._curr_parts[env_index]
            if curr_part:
                eval_parts.append(curr_part)

            # box_state = []
            # ideal_pose_for_curr_part = None
            for part_index in range(NUMBER_PARTS):
                if len(eval_parts) <= part_index:
                    # The worst possible distance is 3m and 180deg
                    self.obs_buf[env_index, (10 + part_index)] = torch.scalar_tensor(3)
                    self.obs_buf[env_index, (10 + NUMBER_PARTS + part_index)] = torch.pi
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

                # Clip obs
                min_dist = min(min_dist, 3)
                rot_dist = min(rot_dist, torch.pi)

                # Record obs
                self.obs_buf[env_index, (10 + part_index)] = min_dist
                self.obs_buf[env_index, (10 + NUMBER_PARTS + part_index)] = rot_dist
                # env_obs['box_state'].append([min_dist, rot_dist])

                if part_index == len(eval_parts) - 1:
                    part_pos_diff = part_pos - torch.tensor(ideal_part[0], device=self._device)
                #     part_rot_euler = R.from_quat(part_rot.cpu()).as_euler('xyz', degrees=False)
                #     ideal_rot_euler = R.from_quat(ideal_part[1]).as_euler('xyz', degrees=False)
                #     part_rot_diff = torch.tensor(ideal_rot_euler - part_rot_euler)
                    self.obs_buf[env_index, 7:10] = part_pos_diff
                #     self.obs_buf[env_index, 10:13] =  part_rot_diff
                #     # env_obs['part_pos_diff'] = part_pos_diff
                #     # env_obs['part_rot_diff'] = part_rot_diff
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
        for env_index in range(self._num_envs):
            if self.reset_buf[env_index]:
                self.reset_env(env_index)
                continue
            
            # Rotate Joints
            robot = self._robots[env_index]
            gripper = robot.gripper

            if not self._curr_parts[env_index]:
                self.reset_robot(env_index)
                self._curr_parts[env_index] = self.add_part(env_index)
                gripper.close()
                continue
            

            # env_step = self.progress_buf[env_index]
            # if env_step == 1:
            #     # We cannot call this in the same step as reset robot since the world needs
            #     # to update once to update the gripper position to the new joint rotations
            #     continue

            joint_rots = robot.get_joint_positions()
            joint_rots += torch.tensor(actions[env_index, 0:6]) * self._max_joint_rot_speed
            robot.set_joint_positions(positions=joint_rots)

            # Open or close Gripper
            is_closed = gripper.is_closed()
            gripper_action = actions[env_index, 6]
            if 0.9 < gripper_action and is_closed:
                gripper.open()
            elif gripper_action < -0.3 and not is_closed:
                gripper.close()

    
    # Calculate Rewards
    # Calculate Rewards
    def calculate_metrics(self) -> None:
        parts_to_ideal_pos = self.obs_buf[:, 7:10]
        targets_dists = torch.linalg.norm(parts_to_ideal_pos, dim=1)

        # self._next_part_buf = targets_dists < 0.07
        next_part_env_indices = (targets_dists < 0.07).nonzero(as_tuple=False).squeeze(-1)
        for env_index in next_part_env_indices:
            self._placed_parts.append(self._curr_parts[env_index])
            self._curr_parts[env_index] = None
            self.progress_buf[env_index] = 0 # A new part gets placed with each reset

        # part_rot_diffs = self.obs_buf[:, 10:13]
        ideal_pos_dists = self.obs_buf[:, 10:(10 + NUMBER_PARTS)]
        ideal_rot_dists = self.obs_buf[:, (10 + NUMBER_PARTS):(10 + 2 * NUMBER_PARTS)]

        box_error_sum = ideal_pos_dists.square().sum(dim=1) + ideal_rot_dists.abs().sum(dim=1)
        self.rew_buf = -targets_dists.square() - box_error_sum

    def is_done(self):
        # any_flipped = False
        self.reset_buf.fill_(0)
        for env_index in range(self._num_envs):
            part = self._curr_parts[env_index]
            if not part:
                continue

            part_pos = part.get_world_pose()[0]

            # Check if part has fallen
            self.reset_buf[env_index] += (part_pos[2] < FALLEN_PART_THRESHOLD - 0.05)

            # if _is_flipped(part_rot):
            #     any_flipped = True
            #     break

        self.reset_buf += (self._max_episode_length - 1 <= self.progress_buf)
        self.reset_buf = self.reset_buf >= 1 # Cast to bool
    
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