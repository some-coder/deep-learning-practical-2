from tensorforce import Agent

class Tensorforce_Agent(object):
    def __init__(self, dyke_1_m, dyke_2_n):
        self.agent = Agent.create(
                agent='tensorforce',
                states=dict(type='float', shape=(dyke_1_m + dyke_2_n,), min_value=0.0),
                actions=dict(type='int', shape=(dyke_1_m + dyke_2_n,), num_values=2),
                memory=10000,
                update=dict(unit='timesteps', batch_size=64),
                optimizer=dict(type='adam', learning_rate=3e-4),
                policy=dict(network='auto'),
                objective='policy_gradient',
                reward_estimation=dict(horizon=20)
                )

