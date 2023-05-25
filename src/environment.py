from gym.wrappers.order_enforcing import OrderEnforcing


class OvercookedHARLSinglePlayerWrapper:
    def __init__(self, env: OrderEnforcing):
        self.env = env
        self.env.env.featurize_fn = lambda x: (x, x)

    def reset(self):
        self.env.reset()
        return None, self.env.base_env.state

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return (None, self.env.base_env.state), reward, done, None