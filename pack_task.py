# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import omni.kit
import torch
from gymnasium import spaces
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import ArticulationView, Articulation
from omni.isaac.core.tasks.base_task import BaseTask
# from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
# from omni.isaac.universal_robots.tasks import PickPlace

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.importer.urdf")
from omni.importer.urdf import _urdf
import omni.kit.commands
import os


class PackTask(BaseTask):
    def __init__(self, name, offset=None) -> None:
        self._robot_position = [0.0, 0.0, 0.0]

        # values used for defining RL buffers
        self._num_observations = 4
        self._num_actions = 1
        self._device = "cpu"
        self.num_envs = 1

        # set the action and observation space for RL
        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0, np.ones(self._num_actions, dtype=np.float32) * 1.0
        )
        
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    # It seems Assets get dowloaded: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0
    # Find a USD File of the Warehouse
    def set_up_scene(self, scene) -> None:
        # super().set_up_scene(scene)
        
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        # env_prim_path="/World/Warehouse"
        # create_prim(prim_path=env_prim_path, prim_type="Xform", position=self._robot_position)
        # usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        # add_reference_to_stage(usd_path, env_prim_path)
        
        # Acquire the URDF extension interface
        urdf_interface = _urdf.acquire_urdf_interface()

        # Set the settings in the import config
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        # import_config.default_drive_strength = 1047.19751
        # import_config.default_position_drive_damping = 52.35988
        # import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import_config.distance_scale = 1
        # import_config.density = 0.0

        # Finally import the robot
        robot_dir = os.getcwd() + '/robot'
        result, robot_prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile", 
                urdf_path=robot_dir + '/ur10e.urdf',
                import_config=import_config
            )
        usd_path = robot_dir + '/ur10e.usd' # assets_root_path + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=robot_prim_path)      
        
        # setting up import configuration:
        # status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        # import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False
        # import_config.import_inertia_tensor = True
        # import_config.fix_base = False
        # import_config.collision_from_visuals = False

        # # Get path to extension data:
        # # ext_manager = omni.kit.app.get_app().get_extension_manager()
        # # ext_id = ext_manager.get_enabled_extension_id("omni.importer.urdf")
        # # extension_path = ext_manager.get_extension_path(ext_id)

        # # import URDF
        # omni.kit.commands.execute(
        #     "URDFParseAndImportFile",
        #     urdf_path=os.getcwd() + "/ur10e.urdf",
        #     import_config=import_config,
        # )
        
        # set default camera viewport position and target
        self.set_initial_camera_params()


    def set_initial_camera_params(self, camera_position=[20, 20, 10], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        return

    def reset(self, env_ids=None):
        return

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self):
        return []

    def calculate_metrics(self) -> None:
        return 0

    def is_done(self) -> None:
        return False