# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import threading
import abc
import asyncio
import queue

import carb
import omni.kit
from omni.isaac.gym.vec_env import VecEnvBase


class TaskStopException(Exception):
    """Exception class for signalling task termination."""

    pass


class TrainerMT(abc.ABC):
    """A base abstract trainer class for controlling starting and stopping of RL policy."""

    @abc.abstractmethod
    def run(self):
        """Runs RL loop in a new thread"""
        pass

    @abc.abstractmethod
    def stop(self):
        """Stop RL thread"""
        pass


class VecEnvMT(VecEnvBase):
    """This class provides a base interface for connecting RL policies with task implementations
    in a multi-threaded fashion. RL policies using this class will run on a different thread
    than the thread simulation runs on. This can be useful for interacting with the UI before,
    during, and after running RL policies. Data sharing between threads happen through message
    passing on multi-threaded queues.
    """

    def initialize(self, action_queue, data_queue, timeout=30):
        """Initializes queues for sharing data across threads.

        Args:
            action_queue (queue.Queue): Queue for passing actions from policy to task.
            data_queue (queue.Queue): Queue for passing data from task to policy.
            timeout (Optional[int]): Seconds to wait for data when queue is empty. An exception will
                                     be thrown when the timeout limit is reached. Defaults to 30 seconds.
        """

        self._action_queue = action_queue
        self._data_queue = data_queue
        self._stop = False
        self._first_frame = True
        self._timeout = timeout

    def get_actions(self, block=True):
        """Retrieves actions from policy by waiting for actions to be sent to the queue from the RL thread.

        Args:
            block (Optional[bool]): Whether to block thread when waiting for data.

        Returns:
            actions (Union[np.ndarray, torch.Tensor, None]): actions buffer retrieved from queue.
        """
        if not self._stop:
            try:
                actions = self._action_queue.get(block, self._timeout)
                if actions is None:
                    self._stop = True
                    self._action_queue.task_done()
                    raise TaskStopException()
                self._action_queue.task_done()
            except (queue.Full, queue.Empty) as e:
                print("Getting actions: timeout occurred.")
                actions = None
                self._stop = True
        else:
            actions = None

        return actions

    def send_actions(self, actions, block=True):
        """Sends actions from RL thread to simulation thread by adding actions to queue.

        Args:
            actions (Union[np.ndarray, torch.Tensor]): actions buffer to be added to queue.
            block (Optional[bool]): Whether to block thread when writing to queue.
        """

        if not self._stop:
            try:
                self._action_queue.put(actions, block, self._timeout)
            except (queue.Full, queue.Empty) as e:
                self._stop = True

    def get_data(self, block=True):
        """Retrieves data from task by waiting for data dictionary to be sent to the queue from the simulation thread.

        Args:
            block (Optional[bool]): Whether to block thread when waiting for data.

        Returns:
            actions (Union[np.ndarray, torch.Tensor, None]): data dictionary retrieved from queue.
        """
        if not self._stop:
            try:
                if self._first_frame:
                    data = self._data_queue.get(block)
                    self._first_frame = False
                else:
                    data = self._data_queue.get(block, self._timeout) # The trainer calls get_data and then waits here until some is available
                if data is None:
                    self._stop = True
                    raise TaskStopException()
                else:
                    self._parse_data(data)
                self._data_queue.task_done()
            except (queue.Full, queue.Empty) as e:
                self._stop = True
                data = None
        else:
            data = None

        return data

    def send_data(self, data, block=True):
        """Sends data from task thread to RL thread by adding data to queue.

        Args:
            data (dict): Dictionary containing task data.
            block (Optional[bool]): Whether to block thread when writing to queue.
        """

        if not self._stop:
            try:
                self._data_queue.put(data, block, self._timeout)
            except (queue.Full, queue.Empty) as e:
                print("Sending states: timeout occurred.")
                self._stop = True

    def clear_queues(self):
        """Clears all queues."""

        while not self._action_queue.empty():
            self._action_queue.get_nowait()
            self._action_queue.task_done()
        while not self._data_queue.empty():
            self._data_queue.get_nowait()
            self._data_queue.task_done()

    def _collect_data(self, obs, rew, reset, extras, states):
        """Helper function to combine buffers into a single dictionary.

        Args:
            obs (Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            rew (Union[numpy.ndarray, torch.Tensor]: Buffer of rewards data.
            reset (Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
            extras (dict): Dictionary of extras data.
            states (Union[numpy.ndarray, torch.Tensor]): Buffer of states data.

        Returns:
            data (dict): Dictionary containing all task buffers.
        """

        data = dict()
        data["obs"] = obs.clone()
        data["rew"] = rew.clone()
        data["reset"] = reset.clone()
        data["extras"] = extras.copy()
        data["states"] = states.clone()

        return data

    def set_render_mode(self, render_mode):
        self._render_mode = render_mode

    async def run(self, trainer):
        """Main loop for controlling simulation and task stepping.
        This method is responsible for stepping task and simulation,
        collecting buffers from task, sending data to policy, and retrieving actions from policy.
        It also deals with the case when the policy terminates on completion and continues
        the simulation thread so that UI does not get affected.

        Args:
            trainer (TrainerMT): A Trainer object that implements APIs for starting and stopping RL thread.
        """
        print('Started isaac sim env loop')

        frames_count = 0
        update_freq = 10
        carb_settings = carb.settings.get_settings()

        self.should_run = True
        while self.should_run:
            try:
                if self._world.is_playing():
                    actions = self.get_actions()
                    if actions is None:
                        continue  # It'll loop until send_action was called
                    await self._task.pre_physics_step_async(actions)
                    self._world._physics_sim_view.flush()
                    for _ in range(self._task.control_frequency_inv - 1):
                        self._world._physics_context._step(current_time=self._world.current_time)
                        self.sim_frame_count += 1

                    # do one step with UI/viewport update
                    if self._render_mode == 0:
                        self.render()
                    else:
                        self._world._physics_context._step(current_time=self._world.current_time)
                    self.sim_frame_count = (self.sim_frame_count + 1) % update_freq

                    obs, rew, reset, extras = await self._task.post_physics_step_async()
                    states = self._task.get_states()
                    data = self._collect_data(obs, rew, reset, extras, states)
                    self.send_data(data) # Send gathered data
                    frames_count = (frames_count + 1) % update_freq

                    # periodically update UI to avoid completely blocking UI
                    if self._render_mode == 2:
                        if frames_count == update_freq - 1:
                            frames_count = 0
                            carb_settings.set_bool("/app/player/playSimulations", False)
                            self.render()
                            carb_settings.set_bool("/app/player/playSimulations", True)
                    elif self._render_mode == 1:
                        carb_settings.set_bool("/app/player/playSimulations", False)
                        self.render()
                        carb_settings.set_bool("/app/player/playSimulations", True)
                elif self._world.is_stopped():
                    self.render()
                    if trainer:
                        # this means simulation was stopped from UI - send stop signal to RL thread
                        self.send_data(None, block=False)
                        self.get_actions(block=False)
                    self.render()
                    break
                else:
                    # i.e. paused - make sure UI is responsive
                    self.render()
                    
            # signals task stopped
            except TaskStopException:
                if trainer:
                    trainer.stop()
                await self._world.stop_async()
                self.render()
                break
            except Exception as e:
                import traceback

                print(traceback.format_exc())
                await self._world.stop_async()
                self.render()
                break


        # task was stopped from RL thread
        if not self.should_run:
            await self._world.stop_async()
            self.render()






















































# FOR DEBUG:

# import threading
# import abc
# import asyncio
# import queue

# import carb
# import omni.kit
# from omni.isaac.gym.vec_env import VecEnvBase


# class TaskStopException(Exception):
#     """Exception class for signalling task termination."""

#     pass


# class TrainerMT(abc.ABC):
#     """A base abstract trainer class for controlling starting and stopping of RL policy."""

#     @abc.abstractmethod
#     def run(self):
#         """Runs RL loop in a new thread"""
#         pass

#     @abc.abstractmethod
#     def stop(self):
#         """Stop RL thread"""
#         pass


# class VecEnvMT(VecEnvBase):
#     """This class provides a base interface for connecting RL policies with task implementations
#     in a multi-threaded fashion. RL policies using this class will run on a different thread
#     than the thread simulation runs on. This can be useful for interacting with the UI before,
#     during, and after running RL policies. Data sharing between threads happen through message
#     passing on multi-threaded queues.
#     """

#     def initialize(self, action_queue, data_queue, timeout=30):
#         """Initializes queues for sharing data across threads.

#         Args:
#             action_queue (queue.Queue): Queue for passing actions from policy to task.
#             data_queue (queue.Queue): Queue for passing data from task to policy.
#             timeout (Optional[int]): Seconds to wait for data when queue is empty. An exception will
#                                      be thrown when the timeout limit is reached. Defaults to 30 seconds.
#         """

#         self._action_queue = action_queue
#         self._data_queue = data_queue
#         self._stop = False
#         self._first_frame = True
#         self._timeout = timeout

#     def get_actions(self, block=True):
#         """Retrieves actions from policy by waiting for actions to be sent to the queue from the RL thread.

#         Args:
#             block (Optional[bool]): Whether to block thread when waiting for data.

#         Returns:
#             actions (Union[np.ndarray, torch.Tensor, None]): actions buffer retrieved from queue.
#         """
#         if not self._stop:
#             try:
#                 actions = self._action_queue.get(block, self._timeout)
#                 print(threading.get_ident(), 'VecEnvMT.get_actions:', actions)
#                 if actions is None:
#                     self._stop = True
#                     print(threading.get_ident(), 'VecEnvMT.get_actions: self._stop = True')
#                     self._action_queue.task_done()
#                     raise TaskStopException()
#                 self._action_queue.task_done()
#             except (queue.Full, queue.Empty) as e:
#                 print("Getting actions: timeout occurred.")
#                 actions = None
#                 self._stop = True
#                 print(threading.get_ident(), 'VecEnvMT.get_actions: self._stop = True')
#         else:
#             actions = None

#         return actions

#     def send_actions(self, actions, block=True):
#         """Sends actions from RL thread to simulation thread by adding actions to queue.

#         Args:
#             actions (Union[np.ndarray, torch.Tensor]): actions buffer to be added to queue.
#             block (Optional[bool]): Whether to block thread when writing to queue.
#         """

#         if not self._stop:
#             try:
#                 self._action_queue.put(actions, block, self._timeout)
#             except (queue.Full, queue.Empty) as e:
#                 print(threading.get_ident(), "Sending actions: timeout occurred.")
#                 self._stop = True
#                 print(threading.get_ident(), 'VecEnvMT.send_actions: self._stop = True')

#     def get_data(self, block=True):
#         """Retrieves data from task by waiting for data dictionary to be sent to the queue from the simulation thread.

#         Args:
#             block (Optional[bool]): Whether to block thread when waiting for data.

#         Returns:
#             actions (Union[np.ndarray, torch.Tensor, None]): data dictionary retrieved from queue.
#         """
#         print(threading.get_ident(), 'VecEnvMT getData 1')
#         if not self._stop:
#             try:
#                 print(threading.get_ident(), 'VecEnvMT getData 1.1')
#                 if self._first_frame:
#                     print(threading.get_ident(), 'VecEnvMT getData 1.2')
#                     data = self._data_queue.get(block)
#                     self._first_frame = False
#                     print(threading.get_ident(), 'VecEnvMT getData 1.2')
#                 else:
#                     print(threading.get_ident(), 'VecEnvMT getData 1.3', list(self._data_queue.queue))
#                     data = self._data_queue.get(block, self._timeout) # The trainer calls get_data and then waits here until some is available
#                     print(threading.get_ident(), 'VecEnvMT getData 1.4')
#                 print(threading.get_ident(), 'VecEnvMT getData 2')
#                 if data is None:
#                     self._stop = True
#                     print(threading.get_ident(), 'VecEnvMT.get_data: self._stop = True')
#                     raise TaskStopException()
#                 else:
#                     self._parse_data(data)
#                 self._data_queue.task_done()
#             except (queue.Full, queue.Empty) as e:
#                 print("Getting states: timeout occurred.")
#                 self._stop = True
#                 print(threading.get_ident(), 'VecEnvMT.get_data2: self._stop = True')
#                 data = None
#         else:
#             print(threading.get_ident(), 'VecEnvMT getData WAS ALREADY STOPPED')
#             data = None

#         return data

#     def send_data(self, data, block=True):
#         """Sends data from task thread to RL thread by adding data to queue.

#         Args:
#             data (dict): Dictionary containing task data.
#             block (Optional[bool]): Whether to block thread when writing to queue.
#         """

#         if not self._stop:
#             try:
#                 self._data_queue.put(data, block, self._timeout)
#             except (queue.Full, queue.Empty) as e:
#                 print("Sending states: timeout occurred.")
#                 self._stop = True
#                 print(threading.get_ident(), 'VecEnvMT.send_data: self._stop = True')

#     def clear_queues(self):
#         """Clears all queues."""

#         while not self._action_queue.empty():
#             self._action_queue.get_nowait()
#             self._action_queue.task_done()
#         while not self._data_queue.empty():
#             self._data_queue.get_nowait()
#             self._data_queue.task_done()

#     def _collect_data(self, obs, rew, reset, extras, states):
#         """Helper function to combine buffers into a single dictionary.

#         Args:
#             obs (Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
#             rew (Union[numpy.ndarray, torch.Tensor]: Buffer of rewards data.
#             reset (Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
#             extras (dict): Dictionary of extras data.
#             states (Union[numpy.ndarray, torch.Tensor]): Buffer of states data.

#         Returns:
#             data (dict): Dictionary containing all task buffers.
#         """

#         data = dict()
#         data["obs"] = obs.clone()
#         data["rew"] = rew.clone()
#         data["reset"] = reset.clone()
#         data["extras"] = extras.copy()
#         data["states"] = states.clone()

#         return data

#     def set_render_mode(self, render_mode):
#         self._render_mode = render_mode

#     async def run(self, trainer):
#         """Main loop for controlling simulation and task stepping.
#         This method is responsible for stepping task and simulation,
#         collecting buffers from task, sending data to policy, and retrieving actions from policy.
#         It also deals with the case when the policy terminates on completion and continues
#         the simulation thread so that UI does not get affected.

#         Args:
#             trainer (TrainerMT): A Trainer object that implements APIs for starting and stopping RL thread.
#         """
#         print('Started isaac sim env loop')

#         frames_count = 0
#         update_freq = 10
#         carb_settings = carb.settings.get_settings()

#         self.should_run = True
#         while self.should_run:
#             print('VecEnvMT actions', list(self.action_queue.queue))
#             print('VecEnvMT data', list(self._data_queue.queue))
#             try:
#                 print(threading.get_ident(), 'VecEnvMT 1')
#                 if self._world.is_playing():
#                     print(threading.get_ident(), 'VecEnvMT 2')
#                     actions = self.get_actions()
#                     if actions is None:
#                         continue  # It'll loop until send_action was called
#                     print(threading.get_ident(), 'VecEnvMT 3')
#                     await self._task.pre_physics_step_async(actions)
#                     print(threading.get_ident(), 'VecEnvMT 3.1')
#                     self._world._physics_sim_view.flush()
#                     for _ in range(self._task.control_frequency_inv - 1):
#                         self._world._physics_context._step(current_time=self._world.current_time)
#                         self.sim_frame_count += 1

#                     # do one step with UI/viewport update
#                     if self._render_mode == 0:
#                         await omni.kit.app.get_app().next_update_async()
#                     else:
#                         self._world._physics_context._step(current_time=self._world.current_time)
#                     self.sim_frame_count = (self.sim_frame_count + 1) % update_freq

#                     print(threading.get_ident(), 'VecEnvMT 4')
#                     obs, rew, reset, extras = await self._task.post_physics_step_async()
#                     print(threading.get_ident(), 'VecEnvMT 5')
#                     states = self._task.get_states()
#                     print(threading.get_ident(), 'VecEnvMT 6')
#                     data = self._collect_data(obs, rew, reset, extras, states)
#                     print(threading.get_ident(), 'VecEnvMT 7')
#                     self.send_data(data) # Send gathered data
#                     print(threading.get_ident(), 'VecEnvMT 8')
#                     frames_count = (frames_count + 1) % update_freq

#                     # periodically update UI to avoid completely blocking UI
#                     if self._render_mode == 2:
#                         print(threading.get_ident(), 'VecEnvMT 9')
#                         if frames_count == update_freq - 1:
#                             frames_count = 0
#                             carb_settings.set_bool("/app/player/playSimulations", False)
#                             print(threading.get_ident(), 'VecEnvMT 10')
#                             await omni.kit.app.get_app().next_update_async() # BUG: GETS STUCK FOREVER IN THIS !!!!
#                             print(threading.get_ident(), 'VecEnvMT 11')
#                             carb_settings.set_bool("/app/player/playSimulations", True)
#                     elif self._render_mode == 1:
#                         carb_settings.set_bool("/app/player/playSimulations", False)
#                         await omni.kit.app.get_app().next_update_async()
#                         carb_settings.set_bool("/app/player/playSimulations", True)
#                 elif self._world.is_stopped():
#                     print(threading.get_ident(), 'VecEnvMT 12')
#                     await omni.kit.app.get_app().next_update_async()
#                     if trainer:
#                         # this means simulation was stopped from UI - send stop signal to RL thread
#                         self.send_data(None, block=False)
#                         self.get_actions(block=False)
#                     await omni.kit.app.get_app().next_update_async()
#                     print(threading.get_ident(), 'VecEnvMT 13')
#                     break
#                 else:
#                     print(threading.get_ident(), 'VecEnvMT 14')
#                     # i.e. paused - make sure UI is responsive
#                     await omni.kit.app.get_app().next_update_async()
#                     print(threading.get_ident(), 'VecEnvMT 15')
                    
#                 print(threading.get_ident(), 'VecEnvMT 16')
#             # signals task stopped
#             except TaskStopException:
#                 print(threading.get_ident(), 'VecEnvMT Task was stopped')
#                 if trainer:
#                     trainer.stop()
#                 await self._world.stop_async()
#                 await omni.kit.app.get_app().next_update_async()
#                 break
#             except Exception as e:
#                 import traceback

#                 print(threading.get_ident(), 'VecEnvMT crashed !!!')
#                 print(traceback.format_exc())
#                 await self._world.stop_async()
#                 await omni.kit.app.get_app().next_update_async()
#                 break

#         print(threading.get_ident(), 'THE EVENT RING WAS BROKEN !!! THE GRACE OF THE GOLDEN ORDER COMPELLS YOU TARNISHED TO FIX IT')

#         # task was stopped from RL thread
#         if not self.should_run:
#             await self._world.stop_async()
#             await omni.kit.app.get_app().next_update_async()