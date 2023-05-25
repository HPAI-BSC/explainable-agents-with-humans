from collections import defaultdict

import numpy as np
from stable_baselines import GAIL
import tensorflow as tf

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.agents.agent import AgentFromPolicy
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import load_pickle


BC_SAVE_DIR = "data/bc_runs/"


def get_agent_from_saved_model(save_dir, sim_threads):
    """Get Agent correspoAgentFromPolicynding to a saved model"""
    # NOTE: Could remove dependency on sim_threads if get the sim_threads from config or dummy env
    state_policy, processed_obs_policy = get_model_policy_from_saved_model(save_dir, sim_threads)
    return AgentFromPolicy(state_policy, processed_obs_policy)


def get_model_policy_from_saved_model(save_dir, sim_threads):
    """Get a policy function from a saved model"""
    predictor = tf.contrib.predictor.from_saved_model(save_dir)
    step_fn = lambda obs: predictor({"obs": obs})["action_probs"]
    return get_model_policy(step_fn, sim_threads)


def get_model_policy(step_fn, sim_threads, is_joint_action=False):
    """
    Returns the policy function `p(s, index)` from a saved model at `save_dir`.

    step_fn: a function that takes in observations and returns the corresponding
             action probabilities of the agent
    """

    def encoded_state_policy(observations, stochastic=True, return_action_probs=False):
        """Takes in SIM_THREADS many losslessly encoded states and returns corresponding actions"""
        action_probs_n = step_fn(observations)

        if return_action_probs:
            return action_probs_n

        if stochastic:
            action_idxs = [np.random.choice(len(Action.ALL_ACTIONS), p=action_probs) for action_probs in action_probs_n]
        else:
            action_idxs = [np.argmax(action_probs) for action_probs in action_probs_n]

        return np.array(action_idxs)

    def state_policy(mdp_state, mdp, agent_index, stochastic=True, return_action_probs=False):
        """Takes in a Overcooked state object and returns the corresponding action"""
        obs = mdp.lossless_state_encoding(mdp_state)[agent_index]
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1))
        action_probs = step_fn(padded_obs)[0]  # Discards all padding predictions

        if return_action_probs:
            return action_probs

        if stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)

        if is_joint_action:
            # NOTE: Probably will break for this case, untested
            action_idxs = Action.INDEX_TO_ACTION_INDEX_PAIRS[action_idx]
            joint_action = [Action.INDEX_TO_ACTION[i] for i in action_idxs]
            return joint_action

        # return Action.INDEX_TO_ACTION[action_idx]
        return action_idx

    return state_policy, encoded_state_policy


def get_bc_agent_from_saved(model_name, no_waits=False):
    model, bc_params = load_bc_model_from_path(model_name)
    return get_bc_agent_from_model(model, bc_params, no_waits), bc_params


def get_bc_agent_from_model(model, bc_params, no_waits=False):
    mdp = OvercookedGridworld.from_layout_name(**bc_params["mdp_params"])
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    def encoded_state_policy(observations, include_waits=True, stochastic=False):
        action_probs_n = model.action_probability(observations)

        if not include_waits:
            action_probs = ImitationAgentFromPolicy.remove_indices_and_renormalize(action_probs_n, [
                Action.ACTION_TO_INDEX[Direction.STAY]])

        if stochastic:
            return [np.random.choice(len(action_probs[i]), p=action_probs[i]) for i in range(len(action_probs))]
        return action_probs_n

    def state_policy(mdp_states, agent_indices, include_waits, stochastic=False):
        # encode_fn = lambda s: mdp.preprocess_observation(s)
        encode_fn = lambda s: mdp.featurize_state(s, mlp)

        obs = []
        for agent_idx, s in zip(agent_indices, mdp_states):
            ob = encode_fn(s)[agent_idx]
            obs.append(ob)
        obs = np.array(obs)
        action_probs = encoded_state_policy(obs, include_waits, stochastic)
        return action_probs

    return ImitationAgentFromPolicy(state_policy, encoded_state_policy, no_waits=no_waits, mlp=mlp)


def load_bc_model_from_path(model_name):
    # NOTE: The lowest loss and highest accuracy models
    # were also saved, can be found in the same dir with
    # special suffixes.
    bc_metadata = load_pickle(BC_SAVE_DIR + model_name + "/bc_metadata")
    bc_params = bc_metadata["bc_params"]
    model = GAIL.load(BC_SAVE_DIR + model_name + "/model")
    return model, bc_params


class ImitationAgentFromPolicy(AgentFromPolicy):
    """Behavior cloning agent interface"""

    def __init__(self, state_policy, direct_policy, mlp=None, stochastic=True, no_waits=False, stuck_time=3):
        super().__init__(state_policy, direct_policy)
        # How many turns in same position to be considered 'stuck'
        self.stuck_time = stuck_time
        self.history_length = stuck_time + 1
        self.stochastic = stochastic
        self.action_probs = False
        self.no_waits = no_waits
        self.will_unblock_if_stuck = False if stuck_time == 0 else True
        self.mlp = mlp
        self.reset()

    def action(self, state):
        return self.actions(state)

    def get_action(self, state):
        return self.action(state.state)

    def actions(self, states, agent_indices=None):
        """
        The standard action function call, that takes in a Overcooked state
        and returns the corresponding action.

        Requires having set self.agent_index and self.mdp
        """
        if agent_indices is None:
            assert isinstance(states, OvercookedState)
            # Chose to overwrite agent index, set it as current agent index. Useful for Vectorized environments
            agent_indices = [self.agent_index]
            states = [states]

        # Actually now state is a list of states
        assert len(states) > 0

        all_actions = self.multi_action(states, agent_indices)

        if len(agent_indices) > 1:
            return all_actions
        return all_actions[0]

    def multi_action(self, states, agent_indices):
        try:
            action_probs_n = list(self.state_policy(states, agent_indices, not self.no_waits))
        except AttributeError:
            raise AttributeError("Need to set the agent_index or mdp of the Agent before using it")

        all_actions = []
        for parallel_agent_idx, curr_agent_action_probs in enumerate(action_probs_n):
            curr_agent_idx = agent_indices[parallel_agent_idx]
            curr_agent_state = states[parallel_agent_idx]
            self.set_agent_index(curr_agent_idx)

            # Removing wait action
            if self.no_waits:
                curr_agent_action_probs = self.remove_indices_and_renormalize(curr_agent_action_probs,
                                                                              [Action.ACTION_TO_INDEX[Direction.STAY]])

            if self.will_unblock_if_stuck:
                curr_agent_action_probs = self.unblock_if_stuck(curr_agent_state, curr_agent_action_probs)

            if self.stochastic:
                action_idx = np.random.choice(len(curr_agent_action_probs), p=curr_agent_action_probs)
            else:
                action_idx = np.argmax(curr_agent_action_probs)
            # curr_agent_action = Action.INDEX_TO_ACTION[action_idx]
            # self.add_to_history(curr_agent_state, curr_agent_action)
            curr_agent_action = action_idx
            self.add_to_history(curr_agent_state, Action.INDEX_TO_ACTION[action_idx])

            if self.action_probs:
                all_actions.append(curr_agent_action_probs)
            else:
                all_actions.append(curr_agent_action)
        return all_actions

    def unblock_if_stuck(self, state, action_probs):
        """Get final action for a single state, given the action probabilities
        returned by the model and the current agent index.
        NOTE: works under the invariance assumption that self.agent_idx is already set
        correctly for the specific parallel agent we are computing unstuck for"""
        stuck, last_actions = self.is_stuck(state)
        if stuck:
            assert any([a not in last_actions for a in Direction.ALL_DIRECTIONS]), last_actions
            last_action_idxes = [Action.ACTION_TO_INDEX[a] for a in last_actions]
            action_probs = self.remove_indices_and_renormalize(action_probs, last_action_idxes)
        return action_probs

    def is_stuck(self, state):
        if None in self.history[self.agent_index]:
            return False, []

        last_states = [s_a[0] for s_a in self.history[self.agent_index][-self.stuck_time:]]
        last_actions = [s_a[1] for s_a in self.history[self.agent_index][-self.stuck_time:]]
        player_states = [s.players[self.agent_index] for s in last_states]
        pos_and_ors = [p.pos_and_or for p in player_states] + [state.players[self.agent_index].pos_and_or]
        if self.checkEqual(pos_and_ors):
            return True, last_actions
        return False, []

    @staticmethod
    def remove_indices_and_renormalize(probs, indices):
        if len(np.array(probs).shape) > 1:
            probs = np.array(probs)
            for row_idx, row in enumerate(indices):
                for idx in indices:
                    probs[row_idx][idx] = 0
            norm_probs = probs.T / np.sum(probs, axis=1)
            return norm_probs.T
        else:
            for idx in indices:
                probs[idx] = 0
            return probs / sum(probs)

    def checkEqual(self, iterator):
        first_pos_and_or = iterator[0]
        for curr_pos_and_or in iterator:
            if curr_pos_and_or[0] != first_pos_and_or[0] or curr_pos_and_or[1] != first_pos_and_or[1]:
                return False
        return True

    def add_to_history(self, state, action):
        assert len(self.history[self.agent_index]) == self.history_length
        self.history[self.agent_index].append((state, action))
        self.history[self.agent_index] = self.history[self.agent_index][1:]

    def reset(self):
        # Matrix of histories, where each index/row corresponds to a specific agent
        self.history = defaultdict(lambda: [None] * self.history_length)

    def update(self, _, __):
        pass