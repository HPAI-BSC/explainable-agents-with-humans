import argparse

from human_aware_rl.human_aware_rl.ppo.ppo import my_config, ppo_run
import tensorflow as tf

if __name__ == '__main__':
    oldinit = tf.Session.__init__


    def myinit(session_object, target='', graph=None, config=None):
        print("Intercepted!")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        oldinit(session_object, target, graph, config)


    tf.Session.__init__ = myinit

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    maps = ['simple', 'unident_s', 'random0', 'random1', 'random3']

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', help='Which map to train in', choices=maps)

    args = parser.parse_args()
    selected_map = args.map

    config = my_config()
    config['params']['mdp_params']['layout_name'] = selected_map

    ppo_run(config['params'])
