import argparse
import pickle

import overcookedgym  # Here to register the env

from src.agent import load_PPO_HARL
from src.discretizers import Discretizer11, Discretizer12, Discretizer13, Discretizer14
from src.environment import OvercookedHARLSinglePlayerWrapper
from src.policy_graph import PolicyGraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discretizer', type=int, help='Which discretizer to use', choices=[11, 12, 13, 14])
    parser.add_argument('--episodes', type=int, help='Amount of episodes to run', default=1500)
    parser.add_argument('--map', help='Which map to create the Policy Graph in',
                        choices=['simple', 'unident_s', 'random0', 'random1', 'random3'])
    parser.add_argument('--normalize', help='Whether the probabilities are stored normalized or not',
                        action='store_true')
    parser.add_argument('--verbose', help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')

    args = parser.parse_args()
    discretizer_id, layout = args.discretizer, args.map
    verbose, normalize = args.verbose, args.normalize

    agent1, agent2, env = load_PPO_HARL(layout)
    wrapped_env = OvercookedHARLSinglePlayerWrapper(env)
    discretizer = {11: Discretizer11, 12: Discretizer12, 13: Discretizer13, 14: Discretizer14}[discretizer_id](env)

    pg = PolicyGraph().fit(agent1, wrapped_env, discretizer, num_episodes=args.episodes, verbose=verbose)

    if normalize:
        with open(f'data/policy_graphs/pg_{layout}_{discretizer_id}_norm.pickle', 'wb') as f:
            pickle.dump(pg.get_normalized_graph(), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'data/policy_graphs/pg_{layout}_{discretizer_id}.pickle', 'wb') as f:
            pickle.dump(pg, f, protocol=pickle.HIGHEST_PROTOCOL)
