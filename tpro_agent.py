from auto_encoder import dyke_auto_encoder
from tensorflow.keras.models import Model
from tensorforce import Agent
from typing import Any, Dict, List, Optional, Union


# data types
LayerSpecification = List[List[Dict[str, any]]]


# constants
AUTO_ENCODER_NODES: int = 5


class TPRO_Agent(object):
    def __init__(self, dyke_1_m, dyke_2_n, L, delta_t, max_episode_timesteps, auto_encoder):
        tfc_policy: Union[Dict[str, Any], LayerSpecification]
        if auto_encoder is not None:
            tfc_policy = \
                    {'network':
                            [
                                    # The special encoding layer. Not trainable.
                                    {
                                            'type': 'keras', 'layer': 'Dense', 'l2_regularization': 0.0,
                                            'units': AUTO_ENCODER_NODES,
                                            'weights': auto_encoder.get_layer(name='code').get_weights(),
                                            'trainable': False
                                    },
                                    # The default layers. Kept as-is here.
                                    {'type': 'dense', 'size': 64, 'activation': 'relu'},
                                    {'type': 'dense', 'size': 64, 'activation': 'relu'},
                                    {'type': 'dense', 'size': 64, 'activation': 'relu'}
                            ]}
        else:
                tfc_policy = {'network': 'auto'}
        self.agent = Agent.create(
                agent='trpo',
                states=dict(type='float', shape=(dyke_1_m + dyke_2_n,), min_value=0.0, max_value=(L+delta_t)),
                actions=dict(type='int', shape=(dyke_1_m + dyke_2_n,), num_values=2),
                max_episode_timesteps = max_episode_timesteps,
                policy=tfc_policy,
                batch_size=64,
                learning_rate=0.001,  # learning rate
                discount=0.99,  # discounting race
                predict_terminal_values=False,  # predict terminal values to know if the episode is almost done
                baseline=dict(type="auto", rnn=False), # critic network
                baseline_optimizer=1.0,
                state_preprocessing=dict(type='instance_normalization'),
                reward_preprocessing=None,
                exploration=0.01  # prob of choosing a random action
	            )

