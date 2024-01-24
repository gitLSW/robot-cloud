import os
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym
from gym_env import GymEnv

MODEL_NAME = "Dreamer"
MAX_STEPS_PER_EPISODE = 300
SIM_STEP_FREQ_HZ = 60

# See configs.yaml for all options.
config = embodied.Config(dreamerv3.configs['defaults'])
config = config.update(dreamerv3.configs['small'])
# config = config.update(dreamerv3.configs['medium'])
# config = config.update(dreamerv3.configs['large'])
#config = config.update(dreamerv3.configs['xlarge'])
config = config.update({
    'logdir': './logdir/' + MODEL_NAME,
    'run.train_ratio': 64,
    'run.log_every': 30,  # Seconds
    'batch_size': 8,
    'batch_length': 16,
    'jax.prealloc': False,
    'encoder.mlp_keys': 'vector',
    'decoder.mlp_keys': 'vector',
    'encoder.cnn_keys': 'image',
    'decoder.cnn_keys': 'image',
    'run.eval_every' : 10000,
    #'jax.platform': 'cpu',
})

config = embodied.Flags(config).parse()
logdir = embodied.Path(config.logdir)
step = embodied.Counter()
logger = embodied.Logger(step, [
    embodied.logger.TerminalOutput(),
    # embodied.logger.TerminalOutput(config.filter),
    embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    embodied.logger.TensorBoardOutput(logdir),
    embodied.logger.WandBOutput(r".*", 'qwertyasd', 'robot-cloud', MODEL_NAME, config),
    # embodied.logger.MLFlowOutput(logdir.name),
])

# Create Isaac environment and open Sim Window
# env = GymEnv(headless=False, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation/manual_livestream_clients.html
env = GymEnv(headless=True, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit', enable_livestream=False)

from pack_task import PackTask # Cannot be imported before Sim has started
task = PackTask(name="Pack", max_steps=MAX_STEPS_PER_EPISODE, sim_s_step_freq=SIM_STEP_FREQ_HZ)
env.set_task(task, backend="numpy", rendering_dt=1 / SIM_STEP_FREQ_HZ)
# env.reset()

env = from_gym.FromGym(env, obs_key='image')
env = dreamerv3.wrap_env(env, config)
env = embodied.BatchEnv([env], parallel=False)

print('Starting Training...')

# env.act_space.discrete = True
# act_space = { 'action': env.act_space }
agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'replay')
args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)
print(args)
embodied.run.train(agent, env, replay, logger, args)

print('Finished Traing')

# env.close()