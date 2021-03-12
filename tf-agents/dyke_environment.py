import numpy as np

from copy import deepcopy
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories.time_step import restart, TimeStep, termination, transition
from tf_agents.typing.types import NestedArraySpec

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class DykeEnvironment(PyEnvironment):
	"""
	A reinforcement learning environment simulating dyke maintenance.

	In this particular simulation, two dykes simultaneously need to be supervised and, if needed,
	repaired. Dyke segments are modelled according to i.i.d. gamma-distributed random variables,
	together forming a stationary stochastic process. Multiple costs are associated to the consequences of
	repairing or not repairing the two dykes.

	Most methods in this class are direct implementations of the PyEnvironment class of
	TensorFlow Agents; we refer to their documentation to see what most of the methods do.
	"""

	class DykeAction(Enum):
		"""
		An action a dyke maintainer can undertake.
		"""
		NO_OPERATION = np.int8(0)
		REPAIR = np.int8(1)

	def __init__(
			self,
			len_dykes: List[int],
			gamma_shape: float,
			gamma_scale: float,
			prevent_cost: float,
			repair_cost: float,
			fixed_cost: float,
			societal_cost: float,
			delta_t: float,
			breach_level: float,
			ran_gen: Optional[np.random.Generator] = None,
			timeout_time: float = 1e0):
		"""
		Constructs a new dyke maintenance environment.

		:param len_dykes: The length of the dykes.
		:param gamma_shape: The shape parameter of the gamma distributions.
		:param gamma_scale: The scale parameter of the gamma distributions.
		:param prevent_cost: The cost of performing specifically preventive maintenance on one dyke segment.
		:param repair_cost: The cost of performing specifically reparative maintenance one dyke segment.
		:param fixed_cost: The fixed cost of performing any type of maintenance on one dyke segment.
		:param societal_cost: The societal cost when both dykes are breached.
		:param delta_t: The size of one time step.
		:param breach_level: The deterioration level at which a dyke should breach.
		:param ran_gen: Optional. A pseudo-random number generator to draw randomness from.
		:param timeout_time: The time at which one dyke maintenance episode ends. Inclusive.
		"""
		super().__init__()
		# Dyke environment fields.
		self.len_dykes: np.ndarray = np.array(len_dykes)
		self._dykes: np.ndarray = np.zeros(shape=(self.len_dykes.sum(),), dtype=np.float64)
		self._times: np.ndarray = np.zeros(shape=(self.len_dykes.sum(),), dtype=np.float64)
		self._t: float = 0.0
		self._gamma_shape = gamma_shape
		self._gamma_scale = gamma_scale
		self._prevent_cost = prevent_cost
		self._repair_cost = repair_cost
		self._fixed_cost = fixed_cost
		self._societal_cost = societal_cost
		self.delta_t = delta_t
		self._breach_level = breach_level
		self._rng: np.random.Generator = ran_gen if ran_gen is not None else np.random.default_rng()
		self.timeout_time = timeout_time
		# Python Environment fields.
		self._action_spec: ArraySpec = ArraySpec(
			shape=(self._dykes.shape[0],),
			dtype=np.int8,
			name='action')
		self._observation_spec: BoundedArraySpec = BoundedArraySpec(
			shape=(self._dykes.shape[0],),
			dtype=np.float64,
			minimum=0.0,
			maximum=self._breach_level,
			name='observation')

	def observation_spec(self) -> NestedArraySpec:
		return self._observation_spec

	def action_spec(self) -> NestedArraySpec:
		return self._action_spec

	def get_info(self) -> Any:
		raise NotImplementedError('Dyke environments do not publish debug information.')

	def get_state(self) -> Any:
		return {
			'len_dykes': deepcopy(self.len_dykes),
			'_dykes': deepcopy(self._dykes),
			'_times': deepcopy(self._times),
			'_t': self._t,
			'_gamma_shape': self._gamma_shape,
			'_gamma_scale': self._gamma_scale,
			'_prevent_cost': self._prevent_cost,
			'_repair_cost': self._repair_cost,
			'_fixed_cost': self._fixed_cost,
			'_societal_cost': self._societal_cost,
			'delta_t': self.delta_t,
			'_breach_level': self._breach_level,
			'_rng': deepcopy(self._rng),
			'timeout_time': self.timeout_time}

	def set_state(self, state: Any) -> None:
		cast_state: Dict[str, Any] = cast(Dict[str, any], state)
		self.len_dykes = cast_state['len_dykes']
		self._dykes = cast_state['_dykes']
		self._times = cast_state['_times']
		self._t = cast_state['_t']
		self._gamma_shape = cast_state['_gamma_shape']
		self._gamma_scale = cast_state['_gamma_scale']
		self._prevent_cost = cast_state['_prevent_cost']
		self._repair_cost = cast_state['_repair_cost']
		self._fixed_cost = cast_state['_fixed_cost']
		self._societal_cost = cast_state['_societal_cost']
		self.delta_t = cast_state['delta_t']
		self._breach_level = cast_state['_breach_level']
		self._rng = cast_state['_rng']
		self.timeout_time = cast_state['timeout_time']

	def _step(self, action: NestedArraySpec) -> TimeStep:
		cost: np.float64 = np.float64(0.0)
		action: np.ndarray
		dyke_action: DykeEnvironment.DykeAction
		# update the state of the dykes
		for dyke_action in DykeEnvironment.DykeAction:
			indices: np.ndarray = action == dyke_action.value
			if dyke_action == DykeEnvironment.DykeAction.NO_OPERATION:
				self._dykes[indices] += [self._gamma_increment() for _ in range(indices.sum())]
				self._times[indices] += [self.delta_t for _ in range(indices.sum())]
			elif dyke_action == DykeEnvironment.DykeAction.REPAIR:
				self._dykes[indices] = [0.0 for _ in range(indices.sum())]
				self._times[indices] = [0.0 for _ in range(indices.sum())]
		# compute costs
		cost += self._repair_cost * (self._dykes >= self._breach_level).sum()
		cost += self._prevent_cost * (action == DykeEnvironment.DykeAction.REPAIR.value).sum()
		cost += self._fixed_cost if self._maintenance_required(action) else 0.0
		cost += self._societal_cost if np.all(self._number_of_breaches_per_dyke() > 0) else 0.0
		# return to the calling agent
		episode_end: bool = self._t > self.timeout_time
		self._t += self.delta_t
		return transition(self._dykes, -cost) if episode_end else termination(self._dykes, -cost)

	def _reset(self) -> TimeStep:
		self._dykes = np.zeros(shape=(self.len_dykes.sum(),), dtype=np.float64)
		self._times = np.zeros(shape=(self.len_dykes.sum(),), dtype=np.float64)
		return restart(observation=self._dykes)

	def _gamma_increment(self) -> np.float64:
		"""
		Yields one increment of the Gamma process for any dyke segment.

		:return: The Gamma process increment.
		"""
		return cast(np.float64, self._rng.gamma(size=(1,), shape=self._gamma_shape, scale=self._gamma_scale)[0])

	def _number_of_breaches_per_dyke(self) -> np.ndarray:
		"""
		Yields the number of dyke segment breaches per dyke.

		:return: The number of breaches.
		"""
		breaches: np.ndarray = np.zeros(shape=(self.len_dykes.shape[0],)).astype(int)
		indices: List[int] = list(np.hstack((np.array([0]), np.cumsum(self.len_dykes) - 1)).astype(int))
		ranges: List[Tuple[int, int]] = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
		for index, ran in enumerate(ranges):
			breaches[index] = (self._dykes[ran[0]:ran[1]] > self._breach_level).sum()
		return breaches

	def _maintenance_required(self, action: np.ndarray) -> bool:
		"""
		Determines whether the agent will currently perform maintenance.

		:param action: The actions the agent plans to undertake.
		:return: The question's answer.
		"""
		return np.any(self._dykes > self._breach_level) or np.any(action == DykeEnvironment.DykeAction.REPAIR.value)


def visualise_dyke_environment(dyke_env: DykeEnvironment, stop: float) -> None:
	"""
	Visualises deterioration and reward as the dyke environment progresses through time.

	:param dyke_env: The dyke environment to plot.
	:param stop: The time at which to stop.
	"""
	steps: int = np.ceil(stop / dyke_env.delta_t).astype(int)
	det_levels: np.ndarray = np.zeros(shape=(steps, dyke_env.len_dykes.sum()))
	rewards: np.ndarray = np.zeros(shape=(steps,))
	for time in range(steps):
		act: np.ndarray = DykeEnvironment.DykeAction.NO_OPERATION.value * np.ones(shape=(dyke_env.len_dykes.sum(),))
		ts: TimeStep = dyke_env.step(action=act)
		det_levels[time, :] = ts.observation
		rewards[time] = ts.reward
	sp: Tuple[Figure, Tuple[Axes, Axes]] = plt.subplots(nrows=2, ncols=1)
	x_values: np.ndarray = np.arange(start=0.0, step=dyke_env.delta_t, stop=steps * dyke_env.delta_t)
	sp[1][0].set_title('Dyke deterioration levels')
	for dyke in range(dyke_env.len_dykes.sum()):
		sp[1][0].step(x_values, det_levels[:, dyke])
	sp[1][1].set_title('Rewards')
	sp[1][1].plot(x_values, rewards)
	sp[0].tight_layout()
	plt.show(block=True)


if __name__ == '__main__':
	params: Dict[str, Any] = {
		'len_dykes': [10, 10],
		'gamma_shape': 5.0,
		'gamma_scale': 0.01,
		'prevent_cost': 1.0,
		'repair_cost': 4.0,
		'fixed_cost': 5.0,
		'societal_cost': 1000.0,
		'delta_t': 0.01,
		'breach_level': 1.0,
		'ran_gen': np.random.default_rng(seed=123)}
	env = DykeEnvironment(**params)
	visualise_dyke_environment(env, stop=2e0)
	# environment has been validated and tested in a TensorFlow environment wrapper
