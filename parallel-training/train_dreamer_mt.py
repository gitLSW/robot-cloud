import os
import dreamerv3
from dreamerv3 import embodied
from embodied.envs import from_gym
from gym_env_mt import GymEnvMT
from functools import partial as bind

name = "test"
MAX_STEPS_PER_EPISODE = 300

# See configs.yaml for all options.
config = embodied.Config(dreamerv3.configs['defaults'])
config = config.update(dreamerv3.configs['small'])
# config = config.update(dreamerv3.configs['medium'])
# config = config.update(dreamerv3.configs['large'])
#config = config.update(dreamerv3.configs['xlarge'])
config = config.update({
    'logdir': './logdir/' + name,
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
    embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    embodied.logger.TensorBoardOutput(logdir),
    #embodied.logger.WandBOutput(r".*",logdir, config),
    # WandBOutputMy(r".*",logdir, config, name),
    # embodied.logger.MLFlowOutput(logdir.name),
])

# Create Isaac environment and open Sim Window
env = GymEnvMT(max_steps = MAX_STEPS_PER_EPISODE,
               sim_s_step_freq = 60,
               headless=False,
               experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')
spacing = 5
offsets = [[spacing, spacing, 0], [spacing, 0, 0], [spacing, -spacing, 0],
           [0, spacing, 0], [0, 0, 0], [0, -spacing, 0],
           [-spacing, spacing, 0], [-spacing, 0, 0], [-spacing, -spacing, 0]]
task_envs = env.init_tasks(offsets, backend="numpy")
# env.reset()

def make_env(task_env):
    task_env = from_gym.FromGym(task_env, obs_key='image')
    task_env = dreamerv3.wrap_env(task_env, config)
    return task_env

ctors = []
for task_env in task_envs:
    ctor = lambda : make_env(task_env)
    ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    ctors.append(ctor)
task_envs = [ctor() for ctor in ctors]
env = embodied.BatchEnv(task_envs, parallel=True)

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