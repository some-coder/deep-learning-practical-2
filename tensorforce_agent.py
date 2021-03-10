from auto_encoder import dyke_auto_encoder
from tensorflow.keras.models import Model
from tensorforce import Agent
from typing import Any, Dict, List, Optional, Union


# data types
LayerSpecification = List[List[Dict[str, any]]]


# constants
AUTO_ENCODER_NODES: int = 5


class Tensorforce_Agent:

	def __init__(self, dyke_1_m: int, dyke_2_n: int, auto_encoder: Optional[Model] = None) -> None:
		tfc_policy: Union[Dict[str, Any], LayerSpecification]
		if auto_encoder is not None:
			tfc_policy = \
				{'network':
					[
						# The special encoding layer. Not trainable.
						{
							'type': 'keras', 'layer': 'Dense', 'l2_regularization': 0.0,
							'units': AUTO_ENCODER_NODES, 'weights': auto_encoder.get_layer(name='code').get_weights(),
							'trainable': False
						},
						# The default layers. Kept as-is here.
						{'type': 'dense', 'size': 64, 'activation': 'tanh'},
						{'type': 'dense', 'size': 64, 'activation': 'tanh'}
					]}
		else:
			tfc_policy = {'network': 'auto'}
		self.agent = Agent.create(
			agent='tensorforce',
			states=dict(type='float', shape=(dyke_1_m + dyke_2_n,), min_value=0.0),
			actions=dict(type='int', shape=(dyke_1_m + dyke_2_n,), num_values=2),
			memory=10000,
			update=dict(unit='timesteps', batch_size=64),
			optimizer=dict(type='adam', learning_rate=3e-4),
			policy=tfc_policy,
			objective='policy_gradient',
			reward_estimation=dict(horizon=20))


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
