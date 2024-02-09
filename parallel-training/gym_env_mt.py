import carb
import uuid
from omni.isaac.gym.vec_env import VecEnvBase
import gymnasium as gym


class GymTaskEnv(gym.Env):
    """
    This class provides a fascade for Gym, so Tasks can be treated like they were envirments.
    """
    def __init__(self, task, gymEnvMT) -> None:
        self._env = gymEnvMT
        self.id = task.name
        self._task = task
        self.observation_space = task.observation_space
        self.action_space = task.action_space

    def reset(self, seed=None, options=None):
        return self._env.reset(self._task, seed)

    def step(self, actions):
        return self._env.step(actions, self._task)



class GymEnvMT(VecEnvBase):
    _tasks = []
    _stepped_tasks = []

    """
    This class handles the Interaction between the different GymTaskEnvs and Isaac
    """
    def __init__(
        self,
        headless: bool,
        sim_device: int = 0,
        enable_livestream: bool = False,
        enable_viewport: bool = False,
        launch_simulation_app: bool = True,
        experience: str = None,
        sim_s_step_freq: float = 60.0,
        max_steps: int = 2000
    ) -> None:
        """Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
            enable_livestream (bool): Whether to enable running with livestream.
            enable_viewport (bool): Whether to enable rendering in headless mode.
            launch_simulation_app (bool): Whether to launch the simulation app (required if launching from python). Defaults to True.
            experience (str): Path to the desired kit app file. Defaults to None, which will automatically choose the most suitable app file.
        """
        self.max_steps = max_steps
        self.rendering_dt = 1 / sim_s_step_freq
        super().__init__(headless, sim_device, enable_livestream, enable_viewport, launch_simulation_app, experience)



    def step(self, actions, task):
        """Basic implementation for stepping simulation.
            Can be overriden by inherited Env classes
            to satisfy requirements of specific RL libraries. This method passes actions to task
            for processing, steps simulation, and computes observations, rewards, and resets.

        Args:
            actions (Union[numpy.ndarray, torch.Tensor]): Actions buffer from policy.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            rewards(Union[numpy.ndarray, torch.Tensor]): Buffer of rewards data.
            dones(Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
            info(dict): Dictionary of extras data.
        """

        if task.name in self._stepped_tasks:
            # Stop thread until all envs have stepped
            raise ValueError(f"Task {task.name} was already stepped in this timestep")
        
        self._stepped_tasks.append(task.name)
        
        task.pre_physics_step(actions)

        if (len(self._stepped_tasks) == len(self._tasks)):
            self._world.step(render=self._render)
            self._stepped_tasks = []
            self.sim_frame_count += 1
            if not self._world.is_playing():
                self.close()

        info = {}
        observations = task.get_observations()
        rewards, done = task.calculate_metrics()
        truncated = done * 0

        return observations, rewards, done, truncated, info



    def reset(self, task, seed=None):
        """Resets the task and updates observations.

        Args:
            seed (Optional[int]): Seed.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            # info(dict): Dictionary of extras data.
        """
        if seed is not None:
            print('RESET GRANDCHILD CLASS UNTESTED')
            seed = self.seed(seed)
            super(GymEnvMT.__bases__[0], self).reset(seed=seed)
        
        obs = task.reset()
        info = {}
        
        # Cannot advance world as resets can happen at any time
        # self._world.step(render=self._render)

        return obs, info # np.zeros(self.observation_space)



    tasks_initialized = False
    def init_tasks(self, offsets, backend="numpy", sim_params=None, init_sim=True) -> None:
        """Creates a World object and adds Task to World.
            Initializes and registers task to the environment interface.
            Triggers task start-up.

        Args:
            task (RLTask): The task to register to the env.
            backend (str): Backend to use for task. Can be "numpy" or "torch". Defaults to "numpy".
            sim_params (dict): Simulation parameters for physics settings. Defaults to None.
            init_sim (Optional[bool]): Automatically starts simulation. Defaults to True.
            rendering_dt (Optional[float]): dt for rendering. Defaults to 1/60s.
        """
        if self.tasks_initialized:
            return [GymTaskEnv(task, self) for task in self._tasks]
        else:
            self.tasks_initialized = True

        from omni.isaac.core.world import World

        # parse device based on sim_param settings
        if sim_params and "sim_device" in sim_params:
            device = sim_params["sim_device"]
        else:
            device = "cpu"
            physics_device_id = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
            gpu_id = 0 if physics_device_id < 0 else physics_device_id
            if sim_params and "use_gpu_pipeline" in sim_params:
                # GPU pipeline must use GPU simulation
                if sim_params["use_gpu_pipeline"]:
                    device = "cuda:" + str(gpu_id)
            elif sim_params and "use_gpu" in sim_params:
                if sim_params["use_gpu"]:
                    device = "cuda:" + str(gpu_id)
                    
        self._world = World(
            stage_units_in_meters=1.0, rendering_dt=self.rendering_dt, backend=backend, sim_params=sim_params, device=device
        )
        # self._world._current_tasks = dict()
        from pack_task_easy import PackTask
        from omni.isaac.core.utils.viewports import set_camera_view

        self._world.scene.add_default_ground_plane()

        for i, offset in enumerate(offsets):
            task = PackTask(f"Task_{i}", self.max_steps, offset, 1 / self.rendering_dt)
            self._world.add_task(task)
            self._tasks.append(task)
        self._num_envs = len(self._tasks)

        first_task = next(iter(self._tasks))
        self.observation_space = first_task.observation_space
        self.action_space = first_task.action_space

        if sim_params and "enable_viewport" in sim_params:
            self._render = sim_params["enable_viewport"]

        if init_sim:
            self._world.reset()
            for task in self._tasks:
                task.reset()
                
        set_camera_view(eye=[-4, -4, 6], target=offsets[len(offsets) - 1], camera_prim_path="/OmniverseKit_Persp")

        return [GymTaskEnv(task, self) for task in self._tasks]



    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True) -> None:
        # Not available for multi task
        raise NotImplementedError()
    


    def update_task_params(self):
        # Not available for multi task
        raise NotImplementedError()