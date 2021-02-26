import numpy as np

from enum import Enum
from typing import Any, cast, Dict, Tuple

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray


class DykeRepairEnv(py_environment.PyEnvironment):
	"""
	Dyke repair reinforcement learning environment.
	"""

	NEVER_REPAIRED: np.int32 = -1

	class Action(Enum):
		"""
		Actions a dyke repair agent can take.
		"""
		NO_OPERATION: np.int32 = np.int32(0)
		REPAIR: np.int32 = np.int32(1)

	def __init__(
			self,
			len_dyke_1: np.int32,
			len_dyke_2: np.int32,
			prevent_cost: np.float64,
			correct_cost: np.float64,
			fixed_cost: np.float64,
			society_cost: np.float64,
			alpha: np.float64,
			beta: np.float64,
			delta_t: np.float64 = np.float64(1e-2),
			threshold: np.float64 = np.float64(1e0),
			run_duration: np.float64 = np.float64(1e0)) -> None:
		"""
		Constructs a new dyke repair reinforcement learning environment.

		@param len_dyke_1: The length of the first dyke.
		@param len_dyke_2: The length of the second dyke.
		@param prevent_cost: The cost of preventing dyke breaches. Non-negative.
		@param correct_cost: The cost of repairing dykes. Non-negative.
		@param fixed_cost: A minimum cost of repair. Non-negative.
		@param society_cost: The societal cost upon a dyke breach. Non-negative.
		@param alpha: The shape parameter for the Gamma process.
		@param beta: The scale parameter for the Gamma process.
		@param delta_t: The timeframe on which to run the experiment. Positive.
		@param threshold: The deterioration level at which a dyke breaches. Non-negative.
		@param run_duration: The length of a run. Positive.
		"""
		super(DykeRepairEnv, self).__init__()
		# specification of the two dykes
		self.len_dyke_1 = len_dyke_1
		self.len_dyke_2 = len_dyke_2
		# set important parameters
		self.prevent_cost = prevent_cost
		self.correct_cost = correct_cost
		self.fixed_cost = fixed_cost
		self.society_cost = society_cost
		self.threshold = threshold
		self.alpha = alpha
		self.beta = beta
		self.delta_t = delta_t
		self.run_duration = run_duration
		self._t: np.float64 = np.float64(0.0)
		self._previous_maintains: np.array = \
			DykeRepairEnv.NEVER_REPAIRED * \
			np.ones(self.len_dyke_1 + self.len_dyke_2, dtype=np.float64)
		# set TensorFlow Agents-related fields
		self._action_spec: array_spec.BoundedArraySpec = \
			array_spec.BoundedArraySpec(
				shape=(1,), dtype=np.int32, minimum=0, maximum=len(DykeRepairEnv.Action) - 1,
				name='action')
		self._observation_spec: array_spec.BoundedArraySpec = \
			array_spec.BoundedArraySpec(
				shape=(self.len_dyke_1 + self.len_dyke_2,), dtype=np.float64, minimum=0.0,
				maximum=self.threshold, name='observation')
		self._state: np.array = np.zeros(self.len_dyke_1 + self.len_dyke_2, dtype=np.float64)
		self._episode_ended: bool = False

	def action_spec(self) -> array_spec.BoundedArraySpec:
		"""
		Defines the actions that should be provided to the `step()` method.

		@return The action specification.
		"""
		return self._action_spec

	def observation_spec(self) -> array_spec.BoundedArraySpec:
		"""
		Defines the observations provided by the environment.

		@return The observations specification.
		"""
		return self._observation_spec

	def _reset(self) -> ts.TimeStep:
		"""
		Starts a new sequence and returns the first `TimeStep` of this sequence.

		@return The first time step of a new episode of the environment.
		"""
		self._state: np.array = np.zeros(self.len_dyke_1 + self.len_dyke_2, dtype=np.float64)
		self._episode_ended = False
		self._previous_maintains: np.array = \
			DykeRepairEnv.NEVER_REPAIRED * np.ones(self.len_dyke_1 + self.len_dyke_2, dtype=np.float64)
		return ts.restart(observation=self._state)

	def _indication(self, maintenance: np.array) -> bool:
		"""
		Determines whether either action is undertaken or is needed.

		@param maintenance: The action taken at this step.
		@return `True` if action is undertaken or needed, `False` otherwise.
		"""
		dyke_1: np.array = self._state[:self.len_dyke_1]
		dyke_2: np.array = self._state[self.len_dyke_1:self.len_dyke_2]
		if dyke_1.max() > self.threshold and dyke_2.max() > self.threshold:
			return True
		elif len(maintenance) > 0:
			return True
		else:
			return False

	def _variable_costs(self, maintenance: np.array) -> np.float64:
		"""
		Computes the preventive and corrective costs of the dyke segments.

		@param maintenance: The targeted dyke segments' deterioration levels.
		@return The variable costs.
		"""
		cost_sum: np.float64 = np.float64(0.0)
		for dyke_segment in maintenance:
			if dyke_segment < self.threshold:
				cost_sum += self.prevent_cost
			elif dyke_segment >= self.threshold:
				cost_sum += self.correct_cost
		return cost_sum

	def _update__previous_maintains(self, indices: np.array) -> None:
		"""
		Updates the 'last maintenance time' array.

		@param indices: Array of indices of dyke patches to address.
		"""
		for index in indices:
			if self._previous_maintains[index] == DykeRepairEnv.NEVER_REPAIRED:
				self._previous_maintains[index] = self._t
			else:
				self._previous_maintains[index] = \
					self._t - self._previous_maintains[index]

	def _should_terminate(self) -> bool:
		return self._t >= self.run_duration

	def _step(self, action: NestedArray) -> ts.TimeStep:
		"""
		Updates the environment according to the action and returns a `TimeStep`.

		@param action: The action taken at this step.
		@return The next time step.
		"""
		# simply return if already done
		if self._episode_ended:
			return self.current_time_step()

		# determine where to apply maintenance
		a: np.array = cast(action, np.array)
		maintenance: np.array = self._state[a == DykeRepairEnv.Action.REPAIR]
		maintenance_indices: np.array = np.where(a == DykeRepairEnv.Action.REPAIR)[0]

		# update previous maintenances, compute cost of actions
		self._update__previous_maintains(maintenance_indices)
		cost: np.float64 = \
			self.fixed_cost if self._indication(maintenance) else 0.0 + \
			self.society_cost if self._indication(maintenance) else 0.0 + \
			self._variable_costs(maintenance)    # TODO: Society term correct?
		if not (self._previous_maintains == DykeRepairEnv.NEVER_REPAIRED).all():
			cost = cost / self._previous_maintains[self._previous_maintains != DykeRepairEnv.NEVER_REPAIRED].mean()

		# update the environment
		self._t += self.delta_t
		new_state: np.array = self._state + \
			np.random.gamma(
				shape=self.alpha * self.delta_t,
				scale=self.beta,
				size=self._state.shape)
		new_state[maintenance_indices] = 0.0
		if self._should_terminate:
			return ts.termination(new_state, reward=-cost)
		else:
			return ts.transition(new_state, reward=-cost)

	def get_info(self) -> Any:
		raise NotImplementedError("The dyke maintenance environment does not yield information.")

	def get_state(self) -> Dict[str, Any]:
		return {
			'time': self._t, 'state': self._state, 'previous': self._previous_maintains,
			'ended': self._episode_ended}

	def set_state(self, state: Dict[str, Any]) -> None:
		keys: Tuple[str, ...] = ('time', 'state', 'previous', 'ended')
		for key in keys:
			if key not in list(state.keys()):
				raise KeyError('Did not find the key %s!' % key)
		self._t = state['time']
		self._state = state['state']
		self._previous_maintains = state['previous']
		self._episode_ended = state['ended']


if __name__ == '__main__':
	environment: py_environment.PyEnvironment = \
		DykeRepairEnv(
			len_dyke_1=np.int32(3),
			len_dyke_2=np.int32(5),
			prevent_cost=np.float64(0.5),
			correct_cost=np.float64(1.5),
			fixed_cost=np.float64(0.1),
			society_cost=np.float64(20.0),
			alpha=np.float64(1.0),
			beta=np.float64(0.5))
	utils.validate_py_environment(environment, episodes=5)
