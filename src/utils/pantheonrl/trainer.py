import gym
from pantheonrl.common.wrappers import frame_wrap, recorder_wrap


def generate_env(args):
    env = gym.make(args.env, **args.env_config)

    altenv = env.getDummyEnv(1)

    if args.framestack > 1:
        env = frame_wrap(env, args.framestack)
        altenv = frame_wrap(altenv, args.framestack)

    if args.record is not None:
        env = recorder_wrap(env)

    return env, altenv
