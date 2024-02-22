import asyncio
import queue
import torch
from skrl import logger

def get_env_instance(headless: bool = True,
                     enable_livestream: bool = False,
                     enable_viewport: bool = False,
                     multi_threaded: bool = False) -> "omni.isaac.gym.vec_env.VecEnvBase":
    """
    Instantiate a VecEnvBase-based object compatible with OmniIsaacGymEnvs

    :param headless: Disable UI when running (default: ``True``)
    :type headless: bool, optional
    :param enable_livestream: Whether to enable live streaming (default: ``False``)
    :type enable_livestream: bool, optional
    :param enable_viewport: Whether to enable viewport (default: ``False``)
    :type enable_viewport: bool, optional
    :param multi_threaded: Whether to return a multi-threaded environment instance (default: ``False``)
    :type multi_threaded: bool, optional

    :return: Environment instance
    :rtype: omni.isaac.gym.vec_env.VecEnvBase

    Example::

        from skrl.envs.wrappers.torch import wrap_env
        from skrl.utils.omniverse_isaacgym_utils import get_env_instance

        # get environment instance
        env = get_env_instance(headless=True)

        # parse sim configuration
        from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
        sim_config = SimConfig({"test": False,
                                "device_id": 0,
                                "headless": True,
                                "multi_gpu": False,
                                "sim_device": "gpu",
                                "enable_livestream": False,
                                "task": {"name": "CustomTask",
                                         "physics_engine": "physx",
                                         "env": {"numEnvs": 512,
                                                 "envSpacing": 1.5,
                                                 "enableDebugVis": False,
                                                 "clipObservations": 1000.0,
                                                 "clipActions": 1.0,
                                                 "controlFrequencyInv": 4},
                                         "sim": {"dt": 0.0083,  # 1 / 120
                                                 "use_gpu_pipeline": True,
                                                 "gravity": [0.0, 0.0, -9.81],
                                                 "add_ground_plane": True,
                                                 "use_flatcache": True,
                                                 "enable_scene_query_support": False,
                                                 "enable_cameras": False,
                                                 "default_physics_material": {"static_friction": 1.0,
                                                                              "dynamic_friction": 1.0,
                                                                              "restitution": 0.0},
                                                 "physx": {"worker_thread_count": 4,
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
                                                           "gpu_max_num_partitions": 8}}}})

        # import and setup custom task
        from custom_task import CustomTask
        task = CustomTask(name="CustomTask", sim_config=sim_config, env=env)
        env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

        # wrap the environment
        env = wrap_env(env, "omniverse-isaacgym")
    """
    from omni.isaac.gym.vec_env import TaskStopException, VecEnvBase, VecEnvMT
    # from omniisaacgymenvs.vec_env_mt_fix import VecEnvMT
    from omni.isaac.gym.vec_env.vec_env_mt import TrainerMT

    class _OmniIsaacGymVecEnv(VecEnvBase):
        def step(self, actions):
            actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()
            self._task.pre_physics_step(actions)

            for _ in range(self._task.control_frequency_inv):
                self._world.step(render=self._render)
                self.sim_frame_count += 1

            observations, rewards, dones, info = self._task.post_physics_step()

            return {"obs": torch.clamp(observations, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()}, \
                rewards.to(self._task.rl_device).clone(), dones.to(self._task.rl_device).clone(), info.copy()

        def reset(self):
            self._task.reset()
            actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
            return self.step(actions)[0]

    class _OmniIsaacGymTrainerMT(TrainerMT):
        def run(self):
            pass

        def stop(self):
            pass

    class _OmniIsaacGymVecEnvMT(VecEnvMT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.set_render_mode(-1 if kwargs['headless'] else 2)

            self.action_queue = queue.Queue(1)
            self.data_queue = queue.Queue(1)

        def run(self, trainer=None):
            asyncio.run(super().run(_OmniIsaacGymTrainerMT() if trainer is None else trainer))

        def _parse_data(self, data):
            self._observations = torch.clamp(data["obs"], -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
            self._rewards = data["rew"].to(self._task.rl_device).clone()
            self._dones = data["reset"].to(self._task.rl_device).clone()
            self._info = data["extras"].copy()

        def step(self, actions):
            if self._stop:
                raise TaskStopException()

            actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).clone()

            self.send_actions(actions) # Send actions
            data = self.get_data() # this waits until data queue has content and then calls _parse_data

            return {"obs": self._observations}, self._rewards, self._dones, self._info

        def reset(self):
            self._task.reset()
            actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
            return self.step(actions)[0]

        def close(self):
            # end stop signal to main thread
            self.send_actions(None)
            self.stop = True

    if multi_threaded:
        try:
            return _OmniIsaacGymVecEnvMT(headless=headless, enable_livestream=enable_livestream, enable_viewport=enable_viewport)
        except TypeError:
            logger.warning("Using an older version of Isaac Sim (2022.2.0 or earlier)")
            return _OmniIsaacGymVecEnvMT(headless=headless)
    else:
        try:
            return _OmniIsaacGymVecEnv(headless=headless, enable_livestream=enable_livestream, enable_viewport=enable_viewport)
        except TypeError:
            logger.warning("Using an older version of Isaac Sim (2022.2.0 or earlier)")
            return _OmniIsaacGymVecEnv(headless=headless)  # Isaac Sim 2022.2.0 and earlier