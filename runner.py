import numpy as np
import os
import pandas as pd
import re
import socket
import time

from agent import Agent, OurProximalPolicyAgent
from environment import Environment
from parameter_grid import Configuration
from typing import cast, List, Optional, Tuple


class Runner:

	SAVE_PATH: Optional[str] = None
	_PEREGRINE_HOST_NAME_INITIALS: str = 'pg-node'
	_PEREGRINE_ARNO_HOME_DIR: str = '/data/s3178471/'
	_AGENT_DIR: str = 'agent'
	_RESULTS_DIR: str = 'results'

	_REWARDS_COLUMNS: Tuple[str, ...] = ('time_step', 'clock_time', 'min_reward', 'quartile_1_reward', 'mean_reward', 'quartile_3_reward', 'max_reward', 'min_cost', 'quartile_1_cost', 'mean_cost', 'quartile_3_cost', 'max_cost')

	@staticmethod
	def _should_set_save_path() -> bool:
		return Runner.SAVE_PATH is None

	@staticmethod
	def _is_run_on_peregrine() -> bool:
		return re.match(pattern=Runner._PEREGRINE_HOST_NAME_INITIALS, string=socket.gethostname()) is not None

	@staticmethod
	def _set_save_path() -> None:
		Runner.SAVE_PATH = Runner._PEREGRINE_ARNO_HOME_DIR if Runner._is_run_on_peregrine() else ''

	@staticmethod
	def _make_save_dir_if_non_existent() -> None:
		paths: Tuple[str, ...] = tuple(Runner.SAVE_PATH + p for p in (Runner._AGENT_DIR, Runner._RESULTS_DIR))
		for path in paths:
			try:
				os.mkdir(path)
			except FileExistsError:
				print('Directory \'%s\' already exists. Skipping...' % (path,))

	@staticmethod
	def _deterioration_df_column_names(m: int, n: int) -> Tuple[str, ...]:
		return tuple(
			['time_step'] +
			['dyke_1_' + str(seg + 1) for seg in range(m)] +
			['dyke_2_' + str(seg + 1) for seg in range(n)])

	@staticmethod
	def _deterioration_df_row(time_step: int, state: List[float]) -> List[float]:
		return [float(time_step)] + state

	@staticmethod
	def _rewards_df_row(time_step: int, clock_time: float, rewards: List[float], real_cost: List[float],) -> List[float]:
		# reward
		rewards_arr: np.ndarray = np.array(rewards)
		quantiles_reward: np.ndarray = np.quantile(a=rewards_arr, q=[0.25, 0.75])
		# cost
		cost_arr: np.ndarray = np.array(real_cost)
		quantiles_cost: np.ndarray = np.quantile(a=cost_arr, q=[0.25, 0.75])
		return [
			float(time_step), clock_time, np.min(rewards_arr), quantiles_reward[0], rewards_arr.mean(), quantiles_reward[1],
			np.max(rewards_arr), np.min(cost_arr), quantiles_cost[0], cost_arr.mean(), quantiles_cost[1],
			np.max(cost_arr), ]

	@staticmethod
	def run(
			env: Environment,
			agn: Agent,
			identifier: str,
			time_steps: int = int(1e4),
			save_agent: bool = False,
			save_interval: Optional[int] = int(1e1),
			info_interval: Optional[int] = int(1e3)) -> None:
		"""
		Conducts a Deep Reinforcement Learning experiment with the supplied environment-agent pair.

		The recommended use of this method is to pair it with ``parameter_grid.configuration_grid``. Grab individual
		configurations from said grid, ``build`` those (yielding an ``Environment`` and ``Agent`` two-tuple), which can
		(almost) directly be supplied to this ``run`` method.

		:param env: An instantiated environment.
		:param agn: An instantiated agent. Must not be an abstract, base-class <tt>agent.Agent</tt>.
		:param identifier: An ID to associate to this experiment. Useful when running multiple experiments.
		:param time_steps: The number of epochs to run the experiment for.
		:param save_agent: Whether to save the agent model. Silently skipped for <tt>agent.NonAgent</tt>.
		:param save_interval: The epoch interval after which data is written to the Pandas data frame.
		:param info_interval: The epoch interval after which information is printed to standard output.
		"""
		# prepare the saving location
		if Runner._should_set_save_path():
			Runner._set_save_path()
			Runner._make_save_dir_if_non_existent()

		# ensure Proximal Policy Agents are configured correctly
		if type(agn) == OurProximalPolicyAgent:
			prx_pol_agn = cast(OurProximalPolicyAgent, agn)  # only used for the ensuing check
			if prx_pol_agn.timeout_time < time_steps:
				msg: str = 'This run\'s OurProximalPolicyAgent can handle up until %d time steps, ' % \
					(prx_pol_agn.timeout_time,)
				msg += 'but the run goes further: up until %d time steps.' % (time_steps,)
				raise ValueError(msg)

		t: float = 0.0  # continuous, environment-bound time
		deterioration_df = pd.DataFrame(columns=Runner._deterioration_df_column_names(env.m, env.n))
		rewards_df = pd.DataFrame(columns=Runner._REWARDS_COLUMNS)

		start_clock_time: float = time.time()
		rewards: List[float] = []
		real_cost: List[float] = []
		for time_step in range(time_steps):  # discrete, runner-bound time
			# conduct the main reinforcement-learning loop
			state = env.observe_state()
			actions = agn.act(states=np.array(state))
			_ = env.take_action(actions=actions)
			cost, reward = env.get_reward()
			real_cost.append(cost)
			rewards.append(reward)
			agn.observe(reward=rewards[-1], terminal=False)
			t += env.delta_t

			if save_interval is not None and time_step % save_interval == 0:
				# update the (wall) clock time
				current_clock_time: float = time.time()
				delta_clock_time: float = current_clock_time - start_clock_time
				start_clock_time = current_clock_time

				# update the data frames
				deterioration_df.loc[len(deterioration_df), :] = \
					Runner._deterioration_df_row(time_step, state)
				rewards_df.loc[len(rewards_df), :] = \
					Runner._rewards_df_row(time_step, delta_clock_time, rewards, real_cost)
				rewards = []  # reset for a new batch
				real_cost = []

			if info_interval is not None and time_step % info_interval == 0:
				print('\t[time step] %5d ' % (time_step,), end='')
				print(f'\t[mean reward & cost] {rewards_df.loc[len(rewards_df) - 1, "mean_reward"]:.3f}, {rewards_df.loc[len(rewards_df) - 1, "mean_cost"]:.3f}')

		if save_agent:
			print('Saving agent... ', end='')
			agn.save(path=Runner.SAVE_PATH + Runner._AGENT_DIR, identifier=identifier)
			print('Done.')

		deterioration_df.convert_dtypes()
		rewards_df.convert_dtypes()
		deterioration_df.to_csv(
			path_or_buf=Runner.SAVE_PATH + Runner._RESULTS_DIR + '/det-' + identifier + '.csv',
			index=False)
		rewards_df.to_csv(
			path_or_buf=Runner.SAVE_PATH + Runner._RESULTS_DIR + '/rew-' + identifier + '.csv',
			index=False)

	@staticmethod
	def save_experiment_identifiers(
			experiment_tag: str,
			configurations: Tuple[Configuration, ...],
			associated_ids: List[int]) -> None:
		df = pd.DataFrame(columns=('id', 'configuration'))
		for index, con in enumerate(configurations):
			df.loc[len(df), :] = [associated_ids[index], str(con)]
		df.to_csv(
			path_or_buf=Runner.SAVE_PATH + Runner._RESULTS_DIR + '/ids-' + experiment_tag + '.csv',
			index=False)
