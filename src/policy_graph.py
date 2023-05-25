from collections import defaultdict
from enum import auto, Enum
from random import choice

from human_aware_rl.utils import set_global_seed
import networkx as nx
import numpy as np


class PolicyGraph(nx.MultiDiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.actions = {}
        self.predicates = {}

    def _run_game(self, agent, environment, discretizer, max_steps=None):
        transition_frequencies = defaultdict(lambda: 0)

        obs = environment.reset()
        done = False

        step_counter = 0
        while not done:
            # We get the action
            action = agent.act(obs)

            previous_obs = obs
            obs, _, done, _ = environment.step(action)

            transition_frequencies[(discretizer.discretize(previous_obs), action, discretizer.discretize(obs))] += 1

            step_counter += 1
            if max_steps is not None and step_counter >= max_steps:
                break

        return transition_frequencies

    def _update(self, frequencies):
        for state_from, action, state_to in frequencies:
            if not self.has_node(state_from):
                self.add_node(state_from, weight=1)
            else:
                self.nodes[state_from]['weight'] += 1

            if not self.has_node(state_to):
                self.add_node(state_to, weight=1)
            else:
                self.nodes[state_to]['weight'] += 1

            if not self.has_edge(state_from, state_to, key=action):
                self.add_edge(state_from, state_to, key=action, weight=1)
            else:
                self[state_from][state_to][action]['weight'] += 1

    def fit(self, agent, environment, discretizer, num_episodes: int = 10, max_steps=None, verbose=False):
        for ep in range(num_episodes):
            if verbose:
                print(f"Episode {ep + 1}/{num_episodes}")
            set_global_seed(ep)
            transition_frequencies = self._run_game(agent, environment, discretizer, max_steps=max_steps)
            self._update(transition_frequencies)

        return self

    def get_normalized_graph(self):
        g = self.copy()

        for node in self.nodes:
            # For each node, get the total sum of the weights of all its edges
            total_frequency = 0
            for dest_node in self[node]:
                for action in self.get_edge_data(node, dest_node):
                    total_frequency += self.get_edge_data(node, dest_node, action)['weight']
            # Normalize the edges with respect to origin node
            for dest_node in self[node]:
                for action in self.get_edge_data(node, dest_node):
                    g.add_edge(node, dest_node, key=action,
                               weight=self.get_edge_data(node, dest_node, action)['weight'] / total_frequency)

        weights = nx.get_node_attributes(self, 'weight')
        total_frequency = sum([weights[k] for k in weights])
        weights2 = {k: weights[k] / total_frequency for k in weights}
        nx.set_node_attributes(g, weights2, 'weight')

        return g


class PGBasedPolicyMode(Enum):
    GREEDY = auto()
    STOCHASTIC = auto()


class PGBasedPolicy:
    def __init__(self, policy_graph: PolicyGraph, discretizer, mode: PGBasedPolicyMode):
        self.pg = policy_graph
        self.pg_norm = self.pg.get_normalized_graph()
        self.discretizer = discretizer
        self.mode = mode

    def _get_nearest_predicate(self, predicate):
        if self.pg.has_node(predicate):
            return predicate
        else:
            dict_predicate = self.discretizer.tuple_to_dict(predicate)
            pred_space = self.discretizer.predicate_space

            for pred_type in pred_space:
                for pred_possible_value in pred_space[pred_type]:
                    candidate_pred = dict_predicate.copy()
                    candidate_pred[pred_type] = pred_possible_value
                    if self.pg.has_node(self.discretizer.dict_to_tuple(candidate_pred)):
                        return self.discretizer.dict_to_tuple(candidate_pred)

        return None

    def _get_action_weights(self, node):
        action_weights = defaultdict(lambda: 0)
        for dest_node in self.pg_norm[node]:
            for action in self.pg_norm[node][dest_node]:
                action_weights[action] += self.pg_norm[node][dest_node][action]['weight']

        action_weights = [(a, action_weights[a]) for a in action_weights]
        return action_weights

    def _get_action(self, action_weights):
        try:
            if self.mode == PGBasedPolicyMode.GREEDY:
                return sorted(action_weights, key=lambda x: x[1], reverse=True)[0][0]
            elif self.mode == PGBasedPolicyMode.STOCHASTIC:
                return np.random.choice([a for a, w in action_weights], p=[w for a, w in action_weights])
        except IndexError:
            print(action_weights)

    def act(self, state):
        predicate = self.discretizer.discretize(state)

        nearest_predicate = self._get_nearest_predicate(predicate)
        if nearest_predicate is None:
            return -1
        else:
            action_weights = self._get_action_weights(nearest_predicate)
            while not action_weights:
                action_weights = self._get_action_weights(choice(list(self.pg_norm.nodes)))

            return self._get_action(action_weights)
