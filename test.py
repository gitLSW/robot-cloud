from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
# from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
# from omni.isaac.core.prims.rigid_prim import RigidPrim

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.universal_robots")
from omni.isaac.universal_robots.ur10 import UR10

import numpy as np

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

robot = UR10(prim_path='/World/UR10', name='UR10')
robot.set_joints_default_state(positions=[-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

world.reset()
robot.initialize()

robot.set_joint_positions(positions=[-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
print(robot.get_joints_default_state())
print(robot.get_joint_positions())

# robot.set_joints_default_state(positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]))
# robot.set_joint_positions ([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

print('World Loaded...')

for i in range(500):
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim