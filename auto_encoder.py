import numpy as np
import tensorflow as tf
from enum import Enum
from environment import Environment as DykeEnvironment
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap


# hyper-parameters
TRAINING_EPOCHS: int = 50
VALIDATION_SCALAR: float = 0.2  # percent of the size of the number of training steps for adding validation steps
CODE_LAYER_ACTIVATION: str = 'relu'
OUTPUT_LAYER_ACTIVATION: str = 'sigmoid'
BATCH_SIZE: int = 64
AUTO_ENCODER_OPTIMIZER: str = 'adam'
AUTO_ENCODER_LOSS_METRIC: str = 'binary_crossentropy'


# other constants
DETERIORATION_LEVEL_FOR_BREACH: float = 1.0
SHOW_TRAINING_STATUS: bool = False
VISUALISATION_SAMPLE_POOL: int = int(1e3)
SHOW_PREDICTION_STATUS: bool = False  # only used during pre-plot prediction


class DykeRepairAction(Enum):
	NO_OPERATION = 0
	REPAIR = 1


def dyke_environment_requires_reset(dyke_env: DykeEnvironment) -> bool:
	for segment in dyke_env.state_1_x_t + dyke_env.state_2_x_t:
		if segment > DETERIORATION_LEVEL_FOR_BREACH:
			return True  # at least one dyke has breached
	return False


def dyke_deterioration_matrix(dyke_env_params: Dict[str, Any], num_samples: int) -> np.array:
	dyke_env = DykeEnvironment(**dyke_env_params)
	matrix: np.array = np.zeros(shape=(num_samples, dyke_env.m + dyke_env.n))
	for t in range(num_samples):
		if dyke_environment_requires_reset(dyke_env):
			dyke_env = DykeEnvironment(**dyke_env_params)
		matrix[t, :] = dyke_env.observe_state()
		dyke_env.take_action(actions=[DykeRepairAction.NO_OPERATION.value] * (dyke_env.m + dyke_env.n))
	return matrix


def dyke_auto_encoder(dyke_env_params: Dict[str, Any], num_samples: int, encoding_size: int) -> tf.keras.models.Model:
	# prepare the deterioration data
	train_steps: int = num_samples
	validation_steps: int = np.floor(train_steps * VALIDATION_SCALAR).astype(int)
	matrix: np.array = dyke_deterioration_matrix(dyke_env_params, train_steps + validation_steps)
	# build the auto-encoder
	in_out_size: int = dyke_env_params['m'] + dyke_env_params['n']
	input_layer = tf.keras.layers.Input(shape=(in_out_size,), name='input')
	code_layer = tf.keras.layers.Dense(units=encoding_size, activation=CODE_LAYER_ACTIVATION, name='code')(input_layer)
	output_layer = tf.keras.layers.Dense(units=in_out_size, activation=OUTPUT_LAYER_ACTIVATION, name='out')(code_layer)
	auto_encoder = tf.keras.models.Model(input_layer, output_layer)
	auto_encoder.compile(optimizer=AUTO_ENCODER_OPTIMIZER, loss=AUTO_ENCODER_LOSS_METRIC)
	# train and return the auto-encoder
	x_train = matrix[:train_steps]
	x_valid = matrix[train_steps:]
	auto_encoder.fit(
		x=x_train, y=x_train, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
		validation_data=(x_valid, x_valid), verbose=1 if SHOW_TRAINING_STATUS else 0)
	return auto_encoder


def visualise_auto_encoder(
		dyke_env_params: Dict[str, Any],
		auto_encoder: tf.keras.models.Model,
		num_samples: int) -> None:
	ins: np.array = dyke_deterioration_matrix(dyke_env_params, num_samples=int(VISUALISATION_SAMPLE_POOL))
	ins = ins[np.random.randint(low=0, high=ins.shape[0], size=num_samples), :]
	outs: np.array = auto_encoder.predict(x=ins, batch_size=BATCH_SIZE, verbose=1 if SHOW_PREDICTION_STATUS else 0)
	sp: Tuple[Figure, Tuple[Axes, Axes]] = plt.subplots(nrows=2, ncols=1)
	sp[0].suptitle('Dyke Deteriorations and Auto Encoder Output')
	for index, axis in enumerate(sp[1]):
		axis.imshow(X=ins if index == 0 else outs, cmap=get_cmap('Greys'))
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)
		axis.set_title('Expected' if index == 0 else 'Actual (Auto-Encoder)')
	plt.show(block=True)


if __name__ == '__main__':
	params: Dict[str, Any] = \
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
	print('Building dyke auto-encoder...', end=' ')
	enc = dyke_auto_encoder(params, num_samples=int(4e4), encoding_size=1)
	print('Done.')
	visualise_auto_encoder(params, enc, num_samples=5)
