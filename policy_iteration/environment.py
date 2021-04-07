import itertools as it
import numpy as np
import scipy.stats as st

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple, Union

from bandit import GaussianBandit


State = Union[int, float, Tuple[int, ...], Tuple[float, ...]]
Action = int
Reward = float
TransitionProbability = float


class Environment(ABC):

	ENV_VAR: int = 1

	class RewardSupplier(ABC):
		"""
		An inner class to compute rewards for the environment.
		"""

		def compute(self, **kwargs) -> float:
			pass

	@abstractmethod
	def __init__(self, **kwargs) -> None:
		pass

	@abstractmethod
	def start(self) -> State:
		pass

	@abstractmethod
	def proceed(self, action: Action) -> Tuple[Reward, State]:
		pass

	@abstractmethod
	def has_stopped(self) -> bool:
		pass

	@property
	@abstractmethod
	def possible_states(self) -> Tuple[State, ...]:
		pass

	@property
	@abstractmethod
	def possible_actions(self) -> Tuple[Action, ...]:
		pass

	@property
	@abstractmethod
	def transition_function(self) -> \
		Dict[Tuple[State, Action], Tuple[Tuple[State, Reward, TransitionProbability], ...]]:
		pass

	@abstractmethod
	def reset(self) -> None:
		pass

	@abstractmethod
	def __str__(self) -> str:
		return 'Abstract environment'


class GaussBanditEnvironment(Environment):

	BanditState: int = 0

	def __init__(self, n: int, mu: float, std: float, noise: float) -> None:
		"""
		Constructs an N-armed bandit environment.

		@param n: The number of bandit arms.
		@param mu: The mean of the bandits' true rewards.
		@param std: The standard deviation of the bandits' true rewards.
		@param noise: The standard deviation of the Gaussian noise around rewards.
		"""
		super(GaussBanditEnvironment, self).__init__()
		self._mu = mu
		self._std = std
		self._noise = noise
		self._rng = np.random.default_rng()
		self._bandits = [GaussianBandit(self._rng.normal(self._mu, self._std), self._noise) for _ in range(n)]

	def start(self) -> State:
		return GaussBanditEnvironment.BanditState,

	def proceed(self, action: Action) -> Tuple[Reward, State]:
		return self._bandits[action].lever_pull_result(), (GaussBanditEnvironment.BanditState,)

	def has_stopped(self) -> bool:
		return False

	@property
	def possible_states(self) -> Tuple[State, ...]:
		return (GaussBanditEnvironment.BanditState,),

	@property
	def possible_actions(self) -> Tuple[Action, ...]:
		return tuple(range(len(self._bandits)))

	@property
	def transition_function(self) -> \
		Dict[Tuple[State, Action], Tuple[Tuple[State, Reward, TransitionProbability], ...]]:
		raise NotImplementedError

	def reset(self) -> None:
		self._bandits = \
			[GaussianBandit(self._rng.normal(self._mu, self._std), self._noise) for _ in range(len(self._bandits))]

	def __str__(self) -> str:
		return 'Gaussian bandit environment'


class StatelessDykeEnvironment(Environment):

	DykeState: int = 0

	class DykeAction(Enum):
		NO_OPERATION = 0
		REPAIR = 1

	class StandardStatelessDykeRewardSupplier(Environment.RewardSupplier):

		def __init__(self, fixed: float, preventive: float, corrective: float, societal: float) -> None:
			self._fixed = fixed
			self._preventive = preventive
			self._corrective = corrective
			self._societal = societal

		def compute(self, action: Action, breach_level: float, dyke: float) -> float:
			rew: float = 0.0
			if action == StatelessDykeEnvironment.DykeAction.REPAIR:
				rew -= self._fixed
				rew -= self._preventive if dyke < breach_level else self._corrective
			return rew

	def __init__(
			self,
			start_level: float,
			breach_level: float,
			reward_supplier: Environment.RewardSupplier,
			gamma_scale: float,
			gamma_shape: float) -> None:
		"""
		Constructs a stateless dyke environment.

		:param start_level: The deterioration level of the dyke at start.
		:param breach_level: The deterioration level at which the dyke breaches.
		:param reward_supplier: An instantiated helper class to aid in computation of rewards.
		:param gamma_scale: The scale parameter of the Gamma distribution.
		:param gamma_shape: The shape parameter of the Gamma distribution.
		"""
		super(StatelessDykeEnvironment, self).__init__()
		self._start_level = start_level
		self._breach_level = breach_level
		self._reward_supplier = reward_supplier
		self._gamma_scl = gamma_scale
		self._gamma_shp = gamma_shape
		self._rng = np.random.default_rng()

	def start(self) -> State:
		return StatelessDykeEnvironment.DykeState,

	def proceed(self, action: Action) -> Tuple[Reward, State]:
		det: float = self._start_level + self._rng.gamma(shape=self._gamma_shp, scale=self._gamma_scl)
		rew: float = self._reward_supplier.compute(action=action, breach_level=self._breach_level, dyke=det)
		return rew, (StatelessDykeEnvironment.DykeState,)

	def has_stopped(self) -> bool:
		return False

	@property
	def possible_states(self) -> Tuple[State, ...]:
		return (StatelessDykeEnvironment.DykeState,),

	@property
	def possible_actions(self) -> Tuple[Action, ...]:
		return tuple([act.value for act in StatelessDykeEnvironment.DykeAction])

	@property
	def transition_function(self) -> \
		Dict[Tuple[State, Action], Tuple[Tuple[State, Reward, TransitionProbability], ...]]:
		raise NotImplementedError

	def reset(self) -> None:
		pass

	def __str__(self) -> str:
		return 'Stateless dyke environment'


class DiscreteDykeEnvironment(Environment):

	NO_OP_ACTION: int = 0  # the no operation action

	@staticmethod
	def action(a: Optional[Action] = None) -> int:
		return DiscreteDykeEnvironment.NO_OP_ACTION if a is None else a + 1

	class ClassicDiscreteDykeRewardSupplier(Environment.RewardSupplier):

		def __init__(self, fixed: float, preventive: float, corrective: float, societal: float) -> None:
			self._fixed = fixed
			self._preventive = preventive
			self._corrective = corrective
			self._societal = societal

		def compute(self, action: Action, breach_level: float, lengths: np.ndarray, dykes: np.ndarray) -> float:
			rew: float = 0.0
			if action != DiscreteDykeEnvironment.NO_OP_ACTION:
				rew -= self._fixed
				rew -= self._preventive if dykes[action - 1] < breach_level else self._corrective
			if DiscreteDykeEnvironment.should_incur_societal_cost(breach_level, lengths, dykes):
				rew -= self._societal
			return rew

	class StandardDiscreteDykeRewardSupplier(ClassicDiscreteDykeRewardSupplier):

		REWARD_BASE: float = 1.005  # base of the exponential term to use for reward calculations

		def __init__(self, fixed: float, preventive: float, corrective: float, societal: float) -> None:
			super(DiscreteDykeEnvironment.StandardDiscreteDykeRewardSupplier, self).__init__(
				fixed, preventive, corrective, societal)
			self._max_reward: Optional[float] = None  # will be initialised late

		def compute(self, action: Action, breach_level: float, lengths: np.ndarray, dykes: np.ndarray) -> float:
			if self._max_reward is None:
				self._max_reward = self._societal + self._fixed + (self._corrective * int(lengths.sum()))
			cost: float = -1.0 * super(DiscreteDykeEnvironment.StandardDiscreteDykeRewardSupplier, self).compute(
				action, breach_level, lengths, dykes)
			return DiscreteDykeEnvironment.StandardDiscreteDykeRewardSupplier.REWARD_BASE ** \
				(self._max_reward - cost)  # we omit the cumulative time component

	def __init__(
			self,
			dyke_lengths: Tuple[int, ...],
			k: int,
			breach_level: float,
			reward_supplier: Environment.RewardSupplier,
			gamma_scale: float,
			gamma_shape: float) -> None:
		super(DiscreteDykeEnvironment, self).__init__()
		self._lengths: np.ndarray = np.array(dyke_lengths, dtype=int)
		self._dykes: np.ndaray = np.zeros(shape=(self._lengths.sum(),))
		self._k = k
		self._breach_level = breach_level
		self._reward_supplier = reward_supplier
		self._gamma_scl = gamma_scale
		self._gamma_shp = gamma_shape
		self._transition_probabilities = self._transition_probabilities_map()
		self._rng = np.random.default_rng()

	def start(self) -> State:
		return tuple(self._dykes)

	def proceed(self, action: Action) -> Tuple[Reward, State]:
		rew: float = self._reward_supplier.compute(
			action=action, breach_level=self._breach_level, lengths=self._lengths, dykes=self._dykes)
		for i in range(self._dykes.shape[0]):
			self._dykes[i] = 0.0 if action - 1 == i else self._updated_dyke_segment(self._dykes[i])
		return rew, tuple(self._dykes)

	def has_stopped(self) -> bool:
		return False

	@property
	def segment_states(self) -> Tuple[float, ...]:
		return tuple(self._breach_level * (lvl / self._k) for lvl in range(self._k + 1))

	@property
	def possible_states(self) -> Tuple[State, ...]:
		return tuple(it.product(self.segment_states, repeat=self._dykes.shape[0]))

	@property
	def possible_actions(self) -> Tuple[Action, ...]:
		return (DiscreteDykeEnvironment.NO_OP_ACTION,) + tuple(range(1, self._lengths.sum() + 1))

	@property
	def transition_function(self) -> \
		Dict[Tuple[State, Action], Tuple[Tuple[State, Reward, TransitionProbability], ...]]:
		d: Dict[Tuple[State, Action], Tuple[Tuple[State, Reward, TransitionProbability], ...]] = {}
		poss_st = self.possible_states
		ac = self.possible_actions
		for s in poss_st:
			for a in ac:
				key: Tuple[State, Action] = (s, a)
				d[key] = tuple()
				for tup in self._possible_next_states(s, a):
					# TODO: Use current or next state for computation of reward? (Currently: next state.)
					rew: float = self._reward_supplier.compute(
						action=a, breach_level=self._breach_level, lengths=self._lengths, dykes=np.array(tup[0]))
					val: Tuple[State, Reward, TransitionProbability] = (tup[0], rew, tup[1])
					d[key] += (val,)
		return d

	def reset(self) -> None:
		pass  # is good as-is

	def __str__(self) -> str:
		return 'Discrete dyke environment'

	def _updated_dyke_segment(self, dyke_segment_state: int) -> float:
		det: float = self._rng.gamma(shape=self._gamma_shp, scale=self._gamma_scl)
		seg_states: Tuple[State, ...] = self.segment_states
		index: int = 0
		while seg_states[index] < dyke_segment_state + det:
			index += 1
		return seg_states[index - 1]

	@staticmethod
	def should_incur_societal_cost(breach_level: float, lengths: np.ndarray, dykes: np.ndarray) -> bool:
		ind: np.ndarray = np.hstack(tup=(np.array([0]), lengths.cumsum()))
		for i in range(ind.shape[0] - 1):
			if np.all(dykes[ind[i]:ind[i + 1]] < breach_level):
				return False  # at least one dyke is fully non-breached
		return True

	def _possible_next_states(self, state: State, action: Action) -> Tuple[Tuple[State, TransitionProbability], ...]:
		next_states: Tuple[Tuple[State, TransitionProbability], ...] = tuple()
		seg_sta = self.segment_states  # to refrain from repeatedly accessing this thing
		per_segment_next_states: Tuple[Tuple[State, ...], ...] = tuple()
		for index, segment in enumerate(state):
			nx_st: Tuple[Tuple[float, ...], ...] = tuple()  # of this dyke segment, their possible next states
			if action - 1 == index:
				nx_st += (0.0,)  # segment will be repaired, so reset to zero deterioration
			else:
				nx_st += seg_sta[seg_sta.index(segment):]  # this state and higher deteriorations
			per_segment_next_states += (nx_st,)
		only_next_states: Tuple[State, ...] = tuple(it.product(*per_segment_next_states))
		for index, ns in enumerate(only_next_states):
			trans_prob: float = self._state_transition_probability(old=state, new=ns, action=action)
			next_states += ((ns, trans_prob),)
		return next_states

	def _transition_probabilities_map(self) -> Dict[Tuple[float, float], float]:
		"""
		Builds a mapping from dyke segment deterioration levels to other levels, and gives their transition chances.

		:return: The map.
		"""
		tpm: Dict[Tuple[float, float], float] = {}
		seg_sta = self.segment_states
		for i in range(len(seg_sta)):
			for j in range(i, len(seg_sta)):  # explicitly include current level
				old_lvl: float = seg_sta[i]
				new_lvl: float = seg_sta[min(j + 1, len(seg_sta) - 1)]
				new_lvl_pre: float = seg_sta[j]
				if j == len(seg_sta) - 1:
					tpm[(old_lvl, new_lvl_pre)] = \
						1 - st.gamma.cdf(x=new_lvl, a=self._gamma_shp, loc=old_lvl, scale=self._gamma_scl)
				else:
					tpm[(old_lvl, new_lvl_pre)] = \
						st.gamma.cdf(x=new_lvl, a=self._gamma_shp, loc=old_lvl, scale=self._gamma_scl) - \
						st.gamma.cdf(x=new_lvl_pre, a=self._gamma_shp, loc=old_lvl, scale=self._gamma_scl)
		return tpm

	def _state_transition_probability(self, old: State, new: State, action: Action) -> float:
		chance: float = 1.0
		for i in range(len(old)):
			if action != DiscreteDykeEnvironment.NO_OP_ACTION and action - 1 == i:
				pass
			else:
				chance *= self._transition_probabilities[(old[i], new[i])]
		return chance
