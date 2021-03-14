import numpy as np
import tensorflow as tf

from dyke_environment import DykeEnvironment, dyke_environment_demo_params
from typing import Any, Dict, List, Optional, Tuple

from tensorflow import Variable
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Constant, RandomUniform, VarianceScaling
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.optimizers import Adam

from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.sequential import Sequential
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.specs.tensor_spec import from_spec
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.utils.common import element_wise_squared_loss, function
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver


# replay buffer settings
_REP_BUF_BATCH_SIZE: int = 4
_REP_BUF_NUM_STEPS: int = 2


def fully_connected_dyke_dqn_agent_network(sizes: Tuple[int, ...]) -> List[Layer]:
	"""
	Constructs a list of DQN agent network layers, all fully connected.

	:param sizes:
	:return:
	"""
	return [Dense(units=size, activation=relu, kernel_initializer=VarianceScaling(scale=2e0)) for size in sizes]


def dyke_dqn_agent(env: TFPyEnvironment, layers: Optional[List[Layer]] = None) -> DqnAgent:
	"""
	Prepares a deep Q-network (DQN) agent for use in the dyke maintenance environment.

	:param env: The dyke environment on which to base the DQN agent.
	:param layers: Optional. A list of layers to supply to the DQN agent's network.
	:return: The agent.
	"""
	layers = fully_connected_dyke_dqn_agent_network(sizes=(100, 50)) if layers is None else layers
	# prepare the Q-values layer
	action_as: BoundedArraySpec = from_spec(env.action_spec())
	number_actions: int = int(action_as.maximum - action_as.minimum + 1)
	q_values_layer: Layer = Dense(
		units=number_actions,
		activation=None,
		kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
		bias_initializer=Constant(-2e-1))
	net = Sequential(layers=layers + [q_values_layer])
	# instantiate and return the agent
	optimizer = Adam(learning_rate=1e-3)
	train_step_counter = Variable(initial_value=0)
	return DqnAgent(
		time_step_spec=env.time_step_spec(),
		action_spec=env.action_spec(),
		q_network=net,
		optimizer=optimizer,
		td_errors_loss_fn=element_wise_squared_loss,
		train_step_counter=train_step_counter)


def _dyke_replay_buffer(env: TFPyEnvironment, agent: DqnAgent, steps_per_episode: int) -> TFUniformReplayBuffer:
	return TFUniformReplayBuffer(
		data_spec=agent.collect_data_spec,
		batch_size=env.batch_size,
		max_length=steps_per_episode)


def _evaluate_dyke_agent(env: TFPyEnvironment, agent: DqnAgent, num_episodes: int = 10) -> np.ndarray:
	returns: np.ndarray = np.zeros(shape=(num_episodes,))
	for ep in range(num_episodes):
		time_step: TimeStep = env.reset()
		episode_return: float = 0.0
		while not time_step.is_last():
			action_step = agent.policy.action(time_step)
			time_step = env.step(action_step.action)
			episode_return += time_step.reward
		returns[ep] = episode_return
	return returns


def train_dyke_agent(
		train_env: TFPyEnvironment,
		eval_env: TFPyEnvironment,
		agent: DqnAgent,
		train_steps: int,
		steps_per_episode: int,
		eval_episodes: int) -> Dict[str, Any]:
	"""
	Trains the DQN agent on the dyke maintenance task.

	:param train_env: The training environment.
	:param eval_env: The environment for testing agent performance.
	:param agent: The agent.
	:param train_steps: The number of training steps to use.
	:param steps_per_episode: The number of time steps that can be taken in a single dyke environment episode.
	:param eval_episodes: The number of episodes to use per evaluation.
	"""
	losses: np.ndarray = np.zeros(shape=(train_steps, steps_per_episode))
	evaluations: np.ndarray = np.zeros(shape=(train_steps, eval_episodes))
	for step in range(train_steps):
		print('Step %d/%d' % (step + 1, train_steps))
		rep_buf = _dyke_replay_buffer(train_env, agent, steps_per_episode)
		obs: Tuple = (rep_buf.add_batch,)
		_ = DynamicEpisodeDriver(
			env=train_env,
			policy=agent.collect_policy,
			observers=obs,
			num_episodes=1).run()  # experience a single episode using the agent's current configuration
		dataset: tf.data.Dataset = rep_buf.as_dataset(
			sample_batch_size=_REP_BUF_BATCH_SIZE,
			num_steps=_REP_BUF_NUM_STEPS)
		iterator = iter(dataset)
		for tr in range(steps_per_episode):
			trajectories, _ = next(iterator)
			losses[step, tr] = agent.train(experience=trajectories).loss
			print('\tAddressed tr = %d/%d' % (tr + 1, steps_per_episode))
		evaluations[step, :] = _evaluate_dyke_agent(eval_env, agent, eval_episodes)
	return {
		'loss-mus': losses.mean(axis=1),
		'loss-sds': losses.std(axis=1),
		'eval-mus': evaluations.mean(axis=1),
		'eval-sds': evaluations.std(axis=1)}


if __name__ == '__main__':
	train_py_env = DykeEnvironment(**dyke_environment_demo_params())
	train_tf_env = TFPyEnvironment(train_py_env)
	eval_py_env = DykeEnvironment(**dyke_environment_demo_params())
	eval_tf_env = TFPyEnvironment(eval_py_env)

	dqn_agent = dyke_dqn_agent(train_tf_env)  # could also have been eval env
	dqn_agent.initialize()

	spe: int = int(np.ceil(train_py_env.timeout_time / train_py_env.delta_t))
	di = train_dyke_agent(train_tf_env, eval_tf_env, dqn_agent, 10, spe, 15)
	print(di['loss-mus'])
	print(di['loss-sds'])
	print(di['eval-mus'])
	print(di['eval-sds'])
