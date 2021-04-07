import numpy as np

from abc import ABC, abstractmethod
from typing import cast, Dict, List, Optional, Tuple

from environment import Action, State


class Agent(ABC):

	@abstractmethod
	def __init__(self, states: Tuple[State, ...], actions: Tuple[Action, ...], **kwargs) -> None:
		self._states = states
		self._actions = actions
		self._times_chosen: Dict[Tuple[State, Action], int] = self._initialised_times_chosen()
		self._vs: Dict[State, float] = self._initialised_vs()
		self._qs: Dict[Tuple[State, Action], float] = self._initialised_qs()
		self._last_state: Optional[State] = None
		self._last_action: Optional[Action] = None

	@abstractmethod
	def decision(self, state: State) -> Action:
		pass

	@abstractmethod
	def experience(self, reward: float) -> None:
		old: float = self._qs[(self._last_state, self._last_action)]
		times: int = self._times_chosen[(self._last_state, self._last_action)]
		self._qs[(self._last_state, self._last_action)] = old + (1 / times) * (reward - old)

	@abstractmethod
	def reset(self) -> None:
		self._times_chosen = self._initialised_times_chosen()
		self._qs = self._initialised_qs()

	@abstractmethod
	def __str__(self) -> str:
		return 'Abstract agent'

	def _initialised_times_chosen(self) -> Dict[Tuple[State, Action], int]:
		d: Dict[Tuple[State, Action], int] = {}
		for s in self._states:
			for a in self._actions:
				d[(s, a)] = 0
		return d

	def _initialised_vs(self) -> Dict[State, float]:
		d: Dict[State, float] = {}
		for s in self._states:
			d[s] = 0.0
		return d

	def _initialised_qs(self) -> Dict[Tuple[State, Action], float]:
		d: Dict[Tuple[State, Action], float] = {}
		for s in self._states:
			for a in self._actions:
				d[(s, a)] = 0.0
		return d


class EpsilonGreedyAgent(Agent):

	def __init__(
			self,
			states: Tuple[State, ...],
			actions: Tuple[Action, ...],
			epsilon: float,
			alpha: Optional[float] = None,
			optimism: float = 0.0) -> None:
		super(EpsilonGreedyAgent, self).__init__(states, actions)
		self._epsilon = epsilon
		self._alpha = alpha
		self._qs = self._initialised_optimistic_qs(optimism)
		self._rng: np.random.Generator = np.random.default_rng()

	def decision(self, state: State) -> Action:
		pick: Action = self._random_action() if self._should_explore() else self._best_action(state)
		self._times_chosen[(state, pick)] += 1
		self._last_state = state
		self._last_action = pick
		return pick

	def experience(self, reward: float) -> None:
		old: float = self._qs[(self._last_state, self._last_action)]
		times: int = self._times_chosen[(self._last_state, self._last_action)]
		self._qs[(self._last_state, self._last_action)] = \
			old + ((1 / times) if self._alpha is None else self._alpha) * (reward - old)

	def reset(self) -> None:
		super(EpsilonGreedyAgent, self).reset()  # is fine as-is

	def __str__(self) -> str:
		return 'Epsilon-Greedy agent'

	def _initialised_optimistic_qs(self, optimism: float) -> Dict[Tuple[State, Action], float]:
		d: Dict[Tuple[State, Action], float] = {}
		for s in self._states:
			for a in self._actions:
				d[(s, a)] = optimism
		return d

	def _should_explore(self) -> bool:
		return self._rng.uniform() < self._epsilon

	def _random_action(self) -> Action:
		return self._rng.choice(self._actions)

	def _best_action(self, state: State) -> Action:
		action_qs: List[float] = []
		for a in self._actions:
			action_qs.append(self._qs[(state, a)])
		return self._actions[int(np.argmax(action_qs))]


class UpperConfidenceBoundAgent(Agent):

	FIRST_TIME_DENOMINATOR: float = 1e-6

	def __init__(self, states: Tuple[State, ...], actions: Tuple[Action, ...], c: float) -> None:
		super(UpperConfidenceBoundAgent, self).__init__(states, actions)
		self._c = c
		self._time: int = 1

	def decision(self, state: State) -> Action:
		pick: Action = self._best_action(state)
		self._times_chosen[(state, pick)] += 1
		self._last_state = state
		self._last_action = pick
		return pick

	def experience(self, reward: float) -> None:
		self._time += 1
		super(UpperConfidenceBoundAgent, self).experience(reward)

	def reset(self) -> None:
		self._time = 1
		super(UpperConfidenceBoundAgent, self).reset()

	def __str__(self) -> str:
		return 'Upper Confidence Bound (UCB) agent'

	def _best_action(self, state: State) -> Action:
		action_qs: List[float] = []
		adapt_qs = self._adapted_qs()
		for a in self._actions:
			action_qs.append(adapt_qs[(state, a)])
		return self._actions[int(np.argmax(action_qs))]

	def _adapted_qs(self) -> Dict[Tuple[State, Action], float]:
		# initialise the adapted Qs mapping
		adapted_qs: Dict[Tuple[State, Action], float] = {}
		for s in self._states:
			for a in self._actions:
				adapted_qs[(s, a)] = 0.0
		# use these to adapt the regular Qs into the adapted Qs with uncertainty bounds
		for key in adapted_qs.keys():
			denominator: float = UpperConfidenceBoundAgent.FIRST_TIME_DENOMINATOR if self._times_chosen[key] == 0 \
				else self._times_chosen[key]
			adapted_qs[key] = self._qs[key] + self._c * np.sqrt(np.log(self._time) / denominator)
		return adapted_qs


class GradientBanditAgent(Agent):

	def __init__(self, states: Tuple[State, ...], actions: Tuple[Action, ...], alpha: float) -> None:
		super(GradientBanditAgent, self).__init__(states, actions)
		self._alpha = alpha
		self._avg_reward: Dict[State, Optional[float]] = self._initialised_avg_reward()
		self._time: int = 1
		self._pmf: np.ndarray = self._updated_pmf()
		self._rng: np.random.Generator = np.random.default_rng()

	def decision(self, state: State) -> Action:
		state_index: int = self._states.index(state)
		pick: Action = self._rng.choice(a=self._actions, p=self._pmf[state_index, :])
		self._times_chosen[(state, pick)] += 1
		self._last_state = state
		self._last_action = pick
		return pick

	def experience(self, reward: float) -> None:
		state_index: int = self._states.index(self._last_state)
		for a in self._actions:
			action_index: int = self._actions.index(a)
			if a == self._last_action:
				self._qs[(self._last_state, a)] += \
					self._alpha * \
					(reward - self._avg_reward[self._last_state]) * \
					(1.0 - self._pmf[state_index, action_index])
			else:
				self._qs[(self._last_state, a)] -= \
					self._alpha * \
					(reward - self._avg_reward[self._last_state]) * \
					self._pmf[state_index, action_index]
		self._update_avg_reward(reward)
		self._time += 1
		self._pmf = self._updated_pmf()

	def reset(self) -> None:
		super(GradientBanditAgent, self).reset()
		self._avg_reward = self._initialised_avg_reward()
		self._time = 1
		self._pmf = self._updated_pmf()

	def __str__(self) -> str:
		return 'Gradient Bandit agent'

	def _initialised_avg_reward(self) -> Dict[State, Optional[float]]:
		d: Dict[State, Optional[float]] = cast(Dict[State, Optional[float]], {})
		for s in self._states:
			d[s] = 0.0
		return d

	def _update_avg_reward(self, reward: float) -> None:
		for s in self._states:
			self._avg_reward[s] = ((0.0 if self._time == 1 else self._avg_reward[s]) + reward) / self._time

	def _updated_pmf(self) -> np.ndarray:
		pmf: np.ndarray = np.zeros(shape=(len(self._states), len(self._actions)))
		for i, s in enumerate(self._states):
			for j, a in enumerate(self._actions):
				pmf[i, j] = np.exp(self._qs[(s, a)])
			pmf[i, :] /= pmf[i, :].sum()  # normalise the SoftMax for this state
		return pmf
