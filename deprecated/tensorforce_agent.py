from auto_encoder import dyke_auto_encoder
from tensorflow.keras.models import Model
from tensorforce import Agent
from typing import Any, Dict, List, Optional, Union


# data types
LayerSpecification = List[List[Dict[str, any]]]


# constants
AUTO_ENCODER_NODES: int = 5


class Tensorforce_Agent:
	def __init__(self, dyke_1_m: int, dyke_2_n: int, L: float, delta_t: float, path):
		self.agent_name = "tensorforce"
		self.saving_frequency = 5 * 60  # save agent every x seconds (real time)
		self.save_agent = True
		self.load_agent = False
		self.network = \
				{'network':
					[  # The default layers. Kept as-is here.
						{'type': 'dense', 'size': 64, 'activation': 'relu'},
						{'type': 'dense', 'size': 64, 'activation': 'relu'},
						{'type': 'dense', 'size': 64, 'activation': 'relu'},
						{'type': 'dense', 'size': 64, 'activation': 'relu'}
					]}
		self.agent = Agent.create(
			agent=self.agent_name,
			states=dict(type='float', shape=(dyke_1_m + dyke_2_n,), min_value=0.0, max_value=(L+delta_t)),
			actions=dict(type='int', shape=(dyke_1_m + dyke_2_n,), num_values=2),
			memory=10000,
			update=dict(unit='timesteps', batch_size=32),
			optimizer=dict(type='adam', learning_rate=3e-4),
			policy=self.network,
			objective='policy_gradient',
			exploration=0.05,  # prob of choosing a random action
			reward_estimation=dict(horizon=(10)), # look at the mean horizon
			saver=dict(directory=path,  # directory path
							   filename=self.agent_name,  # file name for saving
							   frequency=self.saving_frequency,  # save the model every x seconds
							   )
		)

if __name__ == '__main__':
	env_params: Dict[str, Any] = \
		{
			'm': 10,

			'n': 10,
			'alpha': 5.0,
			'beta': 0.01,
			'c_pm': 1,
			'c_cm': 4,
			'c_f': 5,
			'c_s': int(1e3),
			'delta_t': 1e-2,
			'L': 1
		}
	enc: Model = dyke_auto_encoder(env_params, num_samples=int(4e4), encoding_size=5)
	ag = Tensorforce_Agent(10, 10, enc)
