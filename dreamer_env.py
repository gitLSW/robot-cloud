from omni.isaac.gym.vec_env import VecEnvBase


class DreamerEnv(VecEnvBase):
    """This class provides a base interface for connecting RL policies with task implementations.
    APIs provided in this interface follow the interface in gym.Env.
    This class also provides utilities for initializing simulation apps, creating the World,
    and registering a task.
    """

    def __init__(
        self,
        headless: bool,
        sim_device: int = 0,
        enable_livestream: bool = False,
        enable_viewport: bool = False,
        launch_simulation_app: bool = True,
        experience: str = None,
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
        super().__init__(headless, sim_device, enable_livestream, enable_viewport, launch_simulation_app, experience)



    def step(self, actions):
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
        print(self._world.is_playing(), self.sim_frame_count)
        observations, rewards, terminated, truncated, info = super().step(actions)

        if type(rewards) is tuple:
            rewards = rewards[0]
            
        return observations, rewards, terminated, info
    


    def reset(self, seed=None, options=None):
        """Resets the task and updates observations.

        Args:
            seed (Optional[int]): Seed.
            options (Optional[dict]): Options as used in gymnasium.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            info(dict): Dictionary of extras data.
        """
        observations, info = super().reset(seed, options)
        return observations