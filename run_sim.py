import os
from omni.isaac.kit import SimulationApp

sim = SimulationApp({"headless": False}, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

from omni.isaac.core.world import World
world = World(stage_units_in_meters=1.0, backend='numpy')
world.scene.add_default_ground_plane()

from pack_task import PackTask # Cannot be imported before Sim has started
sim_s_step_freq = 60
task = PackTask(name="Pack", max_steps=100000, sim_s_step_freq=sim_s_step_freq)

world.add_task(task)
world.reset()

for step in range(100000):
    world.step(render=True)