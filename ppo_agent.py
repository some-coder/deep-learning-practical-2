from auto_encoder import dyke_auto_encoder
from tensorflow.keras.models import Model
from tensorforce import Agent
from typing import Any, Dict, List, Optional, Union


# data types
LayerSpecification = List[List[Dict[str, any]]]


# constants
AUTO_ENCODER_NODES: int = 5


class PPO_Agent(object):
    def __init__(self, dyke_1_m, dyke_2_n, L, delta_t, max_episode_timesteps, path):
        self.agent_name = "ppo"
        self.saving_frequency = 5 * 60 # save agent every x seconds (real time)
        self.save_agent = True
        self.load_agent = False

        self.network = dict(type='layered', layers=[
                dict(type='dense', size=(dyke_1_m + dyke_2_n), activation='relu'),
                dict(type='dense', size=(dyke_1_m + dyke_2_n), activation='relu'),
                dict(type='lstm', size=(dyke_1_m + dyke_2_n), activation='relu', horizon=10)
            ])
        self.agent = Agent.create(
                agent='ppo',
                states=dict(type='float', shape=(dyke_1_m + dyke_2_n,), min_value=0.0, max_value=(L+delta_t)),
                actions=dict(type='int', shape=(dyke_1_m + dyke_2_n,), num_values=2),
                max_episode_timesteps=max_episode_timesteps,
                batch_size=64,
                learning_rate=0.001,
                network=self.network,
                saver=dict(directory=path,  # directory path
                           filename=self.agent_name,  # file name for saving
                           frequency=self.saving_frequency,  # save the model every x seconds
                           )
	            )



