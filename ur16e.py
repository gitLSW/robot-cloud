

from typing import Optional

import os
from omni.isaac.core.robots.robot import Robot
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot

from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name
enable_extension('omni.isaac.motion_gneration')
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader

class UR16e(Robot):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "ur10_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        attach_gripper (bool, optional): [description]. Defaults to False.
        gripper_usd (Optional[str], optional): [description]. Defaults to "default".

    Raises:
        NotImplementedError: [description]
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur10_robot",
        usd_path: Optional[str] = '/assets/ur16e.usd',
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        gripper_usd: Optional[str] = "/assets/long_gripper.usd",
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        
        # assets_root_path = get_assets_root_path()
        # usd_path = assets_root_path() + "/Isaac/Robots/UniversalRobots/ur16e/ur16e.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        self._end_effector_prim_path = prim_path + "/Gripper"

        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None)

        # gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/long_gripper.usd"
        add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
        self._gripper = SurfaceGripper(
            end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
        )

        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = kinematics_config_dir + "/universal_robots/ur16e/rmpflow/ur16e_robot_description.yaml",
            urdf_path = kinematics_config_dir + "/universal_robots/ur16e/ur16e.urdf"
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")        
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = "gripper"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)

    @property
    def gripper(self) -> SurfaceGripper:
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        self.disable_gravity()
        # self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        # self._end_effector.initialize(physics_sim_view)
        return

    def post_reset(self) -> None:
        Robot.post_reset(self)