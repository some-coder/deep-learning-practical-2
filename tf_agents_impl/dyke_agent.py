import numpy as np
import tensorflow as tf

from dyke_environment import DykeEnvironment, dyke_environment_demo_params
from typing import List, Optional, Tuple

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


# replay buffer settings
_REP_BUF_PREFETCH_NUM: int = 3  # number of time steps to prefetch during DQN training
_REP_BUF_PARALLEL_CALLS: int = 2  # number of elements the replay buffer should process in parallel
_REP_BUF_NUM_STEPS: int = 2  # IMPORTANT. Leave as-is.
_REP_BUF_BATCH_SIZE: int = 64
_REP_BUF_MAX_STEPS: int = int(1e3)


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


def average_return_over_episodes(env: TFPyEnvironment, policy: TFPolicy, num_episodes: int = 10) -> float:
	"""
	Computes the average reward over the specified number of episodes.

	:param env: The environment to assign to the agent.
	:param policy: The policy to supply the agent with.
	:param num_episodes: The number of episodes to average reward over.
	:return: The averaged reward.
	"""
	total_return: np.float64 = np.float64(0.0)
	for i in range(num_episodes):
		time_step: TimeStep = env.reset()
		episode_return: np.float64 = np.float64(0.0)
		t: int = 0
		while not time_step.is_last():
			action_step = policy.action(time_step)
			time_step = env.step(action_step.action)
			episode_return += time_step.reward
			t += 1
		total_return += episode_return
	avg_return = total_return / num_episodes
	return float(avg_return)


def _collect_step(env: TFPyEnvironment, policy: TFPolicy, buffer: TFUniformReplayBuffer) -> None:
	"""
	Saves a single time step to the replay buffer.

	:param env: The environment from which to get the time step.
	:param policy: The policy used by the agent producing the step.
	:param buffer: The buffer to save the step to.
	"""
	time_step: TimeStep = env.current_time_step()
	action_for_step = policy.action(time_step)
	next_time_step: TimeStep = env.step(action_for_step.action)
	trj = from_transition(time_step, action_for_step, next_time_step)
	buffer.add_batch(trj)


def _collect_data(env: TFPyEnvironment, policy: TFPolicy, buffer: TFUniformReplayBuffer, steps: int) -> None:
	"""
	Saves multiple time steps to the replay buffer.

	:param env: The environment from which to get the time step.
	:param policy: The policy used by the agent producing the step.
	:param buffer: The buffer to save the step to.
	:param steps: The number of steps to add to the replay buffer.
	"""
	for _ in range(steps):
		_collect_step(env, policy, buffer)


def train_agent(
		train_env: TFPyEnvironment,
		eval_env: TFPyEnvironment,
		agent: DqnAgent,
		num_pre_eval_episodes: int,
		episodes: int,
		steps_per_episode: int,
		eval_interval: int,
		num_eval_episodes: int) -> List[float]:
	# establish the replay buffer and dataset
	replay_buffer = TFUniformReplayBuffer(
		data_spec=dqn_agent.collect_data_spec,
		batch_size=train_tf_env.batch_size,
		max_length=_REP_BUF_MAX_STEPS)
	dataset: tf.data.Dataset = replay_buffer.as_dataset(
		num_parallel_calls=_REP_BUF_PARALLEL_CALLS,
		sample_batch_size=_REP_BUF_BATCH_SIZE,
		num_steps=_REP_BUF_NUM_STEPS)
	dataset.prefetch(_REP_BUF_PREFETCH_NUM)
	iterator = iter(dataset)
	# train the agent
	agent.train = function(agent.train)  # for optimal performance
	agent.train_step_counter.assign(0)
	print('Computing average return once before training..')
	avg_return = average_return_over_episodes(eval_env, agent.policy, num_episodes=num_pre_eval_episodes)
	returns: List[float] = [avg_return]
	print('Training..')
	for i in range(episodes):
		print('\tIteration %d' % (i,))
		_collect_data(train_env, agent.collect_policy, replay_buffer, steps_per_episode)
		experience, _ = next(iterator)
		_ = agent.train(experience).loss
		step = agent.train_step_counter.numpy()
		if step % eval_interval == 0:
			avg_return = average_return_over_episodes(eval_env, agent.policy, num_eval_episodes)
			# print('[step %d] Average return is %.3lf.' % (step, avg_return))
			returns.append(avg_return)
	return returns


if __name__ == '__main__':
	train_py_env = DykeEnvironment(**dyke_environment_demo_params())
	train_tf_env = TFPyEnvironment(train_py_env)
	eval_py_env = DykeEnvironment(**dyke_environment_demo_params())
	eval_tf_env = TFPyEnvironment(eval_py_env)

	dqn_agent = dyke_dqn_agent(train_tf_env)  # could also have been eval env
	dqn_agent.initialize()

	random_policy = RandomTFPolicy(train_tf_env.time_step_spec(), train_tf_env.action_spec())
	random_policy_return = average_return_over_episodes(train_tf_env, random_policy)
	print('Average random policy return (10 episodes) %.3lf' % (random_policy_return,))
	train_tf_env.reset()

	print('Beginning main program.')
	res = train_agent(
		train_env=train_tf_env,
		eval_env=eval_tf_env,
		agent=dqn_agent,
		num_pre_eval_episodes=5,
		episodes=10,
		steps_per_episode=int(10),
		eval_interval=5,
		num_eval_episodes=100)
	print('Resulting rewards:')
	print(res)
