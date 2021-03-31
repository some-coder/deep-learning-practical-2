import random

from enum import Enum
from typing import List, Tuple


class Environment:

	class RewardFunction(Enum):
		"""
		Stores symbols referencing reward functions to use.
		"""
		STANDARD = 'standard'

	def __init__(
			self,
			m: int,
			n: int,
			alpha: float,
			beta: float,
			c_pm: float,
			c_cm: float,
			c_f: float,
			c_s: float,
			delta_t: float,
			breach_level: float,
			reward_fn: RewardFunction = RewardFunction.STANDARD) -> None:
		"""
		Constructs a new dyke repair reinforcement learning environment.

		:param m: The length of the first dyke.
		:param n: The length of the second dyke.
		:param alpha: The shape parameter for the Gamma process.
		:param beta: The scale parameter for the Gamma process.
		:param c_pm: The cost of preventing dyke breaches. Non-negative.
		:param c_cm: The cost of repairing dykes. Non-negative.
		:param c_f: A minimum cost of repair. Non-negative.
		:param c_s: The societal cost upon a dyke breach. Non-negative.
		:param delta_t: The timeframe on which to run the experiment. Positive.
		:param breach_level: The deterioration level at which a dyke breaches. Non-negative.
		:param reward_fn: The reward function to use.
		"""
		self.m = m
		self.n = n
		self.alpha = alpha
		self.beta = beta
		self.c_pm = c_pm
		self.c_cm = c_cm
		self.c_f = c_f
		self.c_s = c_s
		self.delta_t = delta_t
		self.L = breach_level

		self.max_reward = self.c_s + self.c_f + (self.c_cm * (self.n + self.m))
		self.reward_base = 1.02
		self.risk_taking = 0.5  # most reward at the mean deterioration

		self.state_1_x_t: List[float] = [0.0] * self.m  # first dyke [X(t),...,]
		self.state_2_x_t: List[float] = [0.0] * self.n  # second dyke [X(t),...,]
		self.state_1_t: List[float] = [0.0] * self.m  # first dyke [t,...,]
		self.state_2_t: List[float] = [0.0] * self.n  # second dyke [t,...,]

		self.actions_1 = [0]
		self.actions_2 = [0]
		self.current_cost = 0
		self.current_cum_time = 0

		# set seed
		self.random_generator = random.Random()
		self.random_generator.seed(1234)

		# warm-up state:
		self.warmup_state()

		# reward function
		self.reward_fn = reward_fn

	def warmup_state(self) -> None:
		for i, X_t, in enumerate(self.state_1_x_t):
			self.state_1_x_t[i] = self.random_generator.uniform(0, (self.L+self.delta_t))
		for i, X_t, in enumerate(self.state_2_x_t):
			self.state_2_x_t[i] = self.random_generator.uniform(0, (self.L+self.delta_t))

	# functions for the agent
	def observe_state(self) -> List[float]:
		return self.state_1_x_t + self.state_2_x_t

	def take_action(self, actions) -> bool:
		self.actions_1 = actions[:len(self.state_1_x_t)]
		self.actions_2 = actions[len(self.state_1_x_t):]
		self.update_state()
		return True

	def get_reward(self) -> Tuple[int, float]:
		# print('Our reward function: %s.' % (str(Environment.RewardFunction.STANDARD),))
		if self.reward_fn == Environment.RewardFunction.STANDARD:
			rew = self.reward_base ** (self.max_reward - self.current_cost + self.current_cum_time)
			# print('Reward: %.3lf' % (rew,))
			return self.current_cost, rew
		else:
			raise NotImplementedError('Reward function \'%s\' not implemented (yet).' % (str(self.reward_fn),))

	# internal functions
	def update_state(self) -> None:
		cost = 0
		time = 0

		nr_breaks_first_dyke = 0
		nr_breaks_second_dyke = 0

		maintenance = False

		# first dyke update
		for i, X_t, in enumerate(self.state_1_x_t):
			# control if dyke is breached
			if self.state_1_x_t[i] > self.L:
				nr_breaks_first_dyke += 1

			# perform action
			if self.actions_1[i] == 1: # maintenance
				maintenance = True
				if self.state_1_x_t[i] > self.L:
					cost += self.c_cm  # corrective maintenance
				else:
					cost += self.c_pm  # preventive maintenance
				time += self.state_1_t[i]

				# repair dyke
				self.state_1_x_t[i] = 0
				self.state_1_t[i] = 0

			elif self.actions_1[i] == 0: # do nothing
				# only detoriate breached dyke
				if self.state_1_x_t[i] < self.L:
					self.state_1_x_t[i] += self.gamma_increment()
					self.state_1_t[i] += self.risk_taking
					# avoid overshooting
					if self.state_1_x_t[i] >= self.L:
						self.state_1_x_t[i] = self.L + self.delta_t
				else:
					self.state_1_t[i] += self.risk_taking

		# second dyke update
		for i, X_t, in enumerate(self.state_2_x_t):
			# control if dyke breached
			if self.state_2_x_t[i] > self.L:
				nr_breaks_second_dyke += 1

			# action
			if self.actions_2[i] == 1: # maintenance
				maintenance = True
				if self.state_2_x_t[i] > self.L:
					cost += self.c_cm  # corrective maintenance
				else:
					cost += self.c_pm  # preventive maintenance
				time += self.state_2_t[i]

				# repair dyke
				self.state_2_x_t[i] = 0
				self.state_2_t[i] = 0

			elif self.actions_2[i] == 0:  # do nothing
				# only deteriorate breached dyke
				if self.state_2_x_t[i] < self.L:
					self.state_2_x_t[i] += self.gamma_increment()
					self.state_2_t[i] += self.risk_taking
					# avoid overshooting
					if self.state_2_x_t[i] >= self.L:
						self.state_2_x_t[i] = self.L + self.delta_t
				else:
					self.state_2_t[i] += self.risk_taking

		if maintenance:  # pay fixed cost
			cost += self.c_f

		if nr_breaks_first_dyke > 0 and nr_breaks_second_dyke > 0:  # pay societal cost
			cost += self.c_s

		# update rewards
		self.current_cost = cost
		if time > 0:
			self.current_cum_time = time / (sum(self.actions_1) + sum(self.actions_2)) # mbtf = mean time between failure

	def gamma_increment(self) -> float:
		return self.random_generator.gammavariate(alpha=(self.alpha * self.delta_t),beta=self.beta)
