import argparse
import csv
import pickle

import overcookedgym  # Here to register the env

from human_aware_rl.human_aware_rl.utils import set_global_seed
from src.agent import load_PPO_HARL
from src.discretizers import Discretizer11, Discretizer12, Discretizer13, Discretizer14
from src.environment import OvercookedHARLSinglePlayerWrapper
from src.policy_graph import PGBasedPolicy, PGBasedPolicyMode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discretizer', type=int, help='Which discretizer to use',
                        choices=[11, 12, 13, 14], default=11)
    parser.add_argument('--episodes', type=int, help='Amount of episodes to run', default=500)
    parser.add_argument('--map', help='Which map to create the Policy Graph in',
                        choices=['simple', 'unident_s', 'random0', 'random1', 'random3'])
    parser.add_argument('--policy-mode',
                        help='Whether to use the original agent, or a greedy or stochastic PG-based policy',
                        choices=['original', 'greedy', 'stochastic'])


    args = parser.parse_args()
    discretizer_id, layout, episodes = args.discretizer, args.map, args.episodes

    agent1, agent2, env = load_PPO_HARL(layout)

    if args.policy_mode == 'original':
        agent = agent1
    else:
        discretizer = {11: Discretizer11, 12: Discretizer12, 13: Discretizer13, 14: Discretizer14}[discretizer_id](env)

        if args.policy_mode == 'greedy':
            mode = PGBasedPolicyMode.GREEDY
        else:
            mode = PGBasedPolicyMode.STOCHASTIC

        with open(f'data/policy_graphs/pg_{layout}_{discretizer_id}.pickle', 'rb+') as f:
            pg = pickle.load(f)
            agent = PGBasedPolicy(pg, discretizer, mode)

    wrapped_env = OvercookedHARLSinglePlayerWrapper(env)

    obtained_rewards = []
    for ep in range(episodes):
        set_global_seed(ep)

        episode_reward = 0

        obs = wrapped_env.reset()
        done = False
        while not done:
            # We get the action
            action = agent.act(obs)

            obs, reward, done, _ = wrapped_env.step(action)

            episode_reward += reward

        obtained_rewards.append(episode_reward)

    with open(f'data/rewards/rewards_{layout}_{args.policy_mode}{"" if args.policy_mode == "original" else f"_{discretizer_id}"}.csv', 'w+') as f:
        csv_w = csv.writer(f)
        csv_w.writerow(obtained_rewards)
