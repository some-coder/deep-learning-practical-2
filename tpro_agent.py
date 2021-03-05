from tensorforce import Agent

class TPRO_Agent(object):
    def __init__(self, dyke_1_m, dyke_2_n, max_episode_timesteps):
        self.agent = Agent.create(
                agent='trpo',
                states=dict(type='float', shape=(dyke_1_m + dyke_2_n,), min_value=0.0),
                actions=dict(type='int', shape=(dyke_1_m + dyke_2_n,), num_values=2),
                max_episode_timesteps = max_episode_timesteps,
                network=dict(type="auto", rnn=False),
                batch_size=64,
                learning_rate=0.01,  # learning rate
                discount=0.99,  # discounting race
                predict_terminal_values=False,  # predict terminal values to know if the episode is almost done
                baseline=dict(type="auto", rnn=False), # critic network
                baseline_optimizer=1.0,
                state_preprocessing=dict(type='instance_normalization'),
                reward_preprocessing=None,
                exploration=0.01,  # prob of choosing a random action
                variable_noise=0.0,
                l2_regularization=0.01,
                entropy_regularization=0.001
	            )

