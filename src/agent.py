import os.path
import pickle
from argparse import Namespace

import numpy as np

import pantheonrl.common.util as prl
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from src.utils.human_aware_rl.baselines_utils import get_agent_from_saved_model, get_bc_agent_from_saved
from src.utils.pantheonrl.trainer import generate_env


class OvercookedAgent:
    def __init__(self, agent):
        self.agent = agent

    def act(self, obs):
        return self.agent.action(obs[1])

    def act_batch(self, obs):
        actions, _, _ = prl.action_from_policy(np.array([o[0] for o in obs]), self.agent.policy)
        return prl.clip_actions(actions, self.agent.policy)


def load_PPO_HARL(layout):
    params = Namespace(
        env='OvercookedMultiEnv-v0',
        env_config={'layout_name': layout},
        framestack=1,
        record=None
    )
    env, altenv = generate_env(params)

    if os.path.exists(f"data/ppo_runs/{layout}_agent/seed0/best"):
        ego_path = f"data/ppo_runs/{layout}_agent/seed0/best"
    else:
        ego_path = f"data/ppo_runs/{layout}_agent/seed0/ppo_agent"
    ego = OvercookedAgent(get_agent_from_saved_model(ego_path, 30))
    ego.agent.set_agent_index(0)
    ego.agent.set_mdp(OvercookedGridworld.from_layout_name(layout))

    with open("data/bc_runs/best_bc_model_paths.pickle", "rb") as f:
        best = pickle.load(f)

    alt, _ = get_bc_agent_from_saved(best['train'][layout])
    alt.set_agent_index(1)

    env.add_partner_agent(alt)

    return ego, alt, env
