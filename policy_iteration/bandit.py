import numpy as np

from abc import ABC, abstractmethod


class Bandit(ABC):

	@abstractmethod
	def __init__(self) -> None:
		pass

	@abstractmethod
	def lever_pull_result(self) -> float:
		pass


class GaussianBandit(Bandit):

	def __init__(self, true_reward: float, noise: float) -> None:
		super(GaussianBandit, self).__init__()
		self._true_reward = true_reward
		self._noise = noise

	def lever_pull_result(self) -> float:
		return np.random.normal(loc=self._true_reward, scale=self._noise)
