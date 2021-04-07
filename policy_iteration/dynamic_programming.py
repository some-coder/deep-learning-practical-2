import numpy as np

from environment import Action, Environment, Reward, State, TransitionProbability
from typing import Dict, List


class PolicyIterator:

	ALLOWABLE_EVALUATION_ERROR: float = 1e-6

	def __init__(self, env: Environment, discount: float) -> None:
		self._states = env.possible_states
		self._actions = env.possible_actions
		self.vs = np.zeros(shape=(len(self._states),))
		self._rng = np.random.default_rng()  # must precede the declaration of the policy
		self.policy: Dict[State, Action] = self._arbitrary_policy()
		self._discount = discount  # also known as lambda
		self._policy_is_stable: bool = False
		self._transitions = env.transition_function

	def _evaluate_policy(self) -> None:
		while True:
			delta: float = 0.0
			for index, s in enumerate(self._states):
				v: float = self.vs[index]
				self.vs[index] = self._updated_state_value(state_index=index)
				delta = np.max((delta, np.abs(v - self.vs[index])))
			if delta < PolicyIterator.ALLOWABLE_EVALUATION_ERROR:
				break

	def _improve_policy(self) -> None:
		self._policy_is_stable = True
		for index, s in enumerate(self._states):
			old_action: Action = self.policy[s]
			self.policy[s] = self._updated_state_action(state_index=index)
			self._policy_is_stable = False if old_action != self.policy[s] else True

	def iterate(self) -> None:
		i: int = 0
		while not self._policy_is_stable:
			print('Iteration %d.' % (i + 1,))
			self._evaluate_policy()
			self._improve_policy()
			i += 1

	def _arbitrary_policy(self) -> Dict[State, Action]:
		d: Dict[State, Action] = {}
		for s in self._states:
			d[s] = self._rng.choice(self._actions)
		return d

	def _updated_state_value(self, state_index: int) -> float:
		state: State = self._states[state_index]
		return self._new_state_action_value(state=state, action=self.policy[state])

	def _updated_state_action(self, state_index: int) -> Action:
		state: State = self._states[state_index]
		qs: List[float] = [self._new_state_action_value(state, a) for a in self._actions]
		return self._actions[int(np.argmax(qs))]

	def _new_state_action_value(self, state: State, action: Action) -> float:
		v: float = 0.0
		for tup in self._transitions[(state, action)]:
			next_state: State = tup[0]
			next_state_index: int = self._states.index(next_state)
			reward: Reward = tup[1]
			trans_prob: TransitionProbability = tup[2]
			v += trans_prob * (reward + self._discount * self.vs[next_state_index])
		return v
