from random import Random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

from tensorforce import Agent as TensorForceAgent  # Not to be confused with Tensor(f)orceAgent.


Action = int
AgentParameterMap = Dict[str, Any]
TensorForceModel = Dict[str, List[Dict[str, Any]]]  # this is how we define networks in TensorForce


class Agent(ABC):

	NAME: str = 'abstract-agent'

	@abstractmethod
	def __init__(self, m: int, n: int, **kwargs) -> None:
		self._m = m
		self._n = n

	@abstractmethod
	def act(self, states: Any) -> List[Action]:
		pass

	@abstractmethod
	def observe(self, reward: float, terminal: bool) -> int:
		pass

	@abstractmethod
	def save(self, path: str, identifier: str) -> None:
		pass


AgentParameterMapPair = Tuple[Type[Agent], AgentParameterMap]  # We must define this type after Agent.


class OurTensorForceAgent(Agent):

	NAME: str = 'tensor-force-agent'

	_SPECIFICATION_KEY: str = 'tensorforce'
	_MEMORY: int = int(1e4)
	_NUM_ACTIONS: int = 2
	_BATCH_SIZE: int = 32
	_OPTIMIZER_NAME: str = 'adam'
	_OPTIMIZER_LEARNING_RATE: float = 3e-4
	_OPTIMIZER_GRADIENT_CLIP_THRESHOLD: float = 1e-2
	_OBJECTIVE: str = 'policy_gradient'
	_EXPLORATION_RATE: float = 5e-2
	_REWARD_HORIZON: int = 10

	_SAVE: bool = False
	_SAVE_NAME: str = NAME
	_SAVING_FREQUENCY: int = 5 * 60

	def __init__(
			self,
			m: int,
			n: int,
			breach_level: float,
			delta_t: float,
			learning_rate: float,
			use_gradient_clipping: bool,
			save_path: str,
			model: TensorForceModel) -> None:
		super(OurTensorForceAgent, self).__init__(m, n)
		self._breach_level = breach_level
		self._delta_t = delta_t
		self._save_path = save_path
		self._model = model
		self._tensor_force_agent: TensorForceAgent = TensorForceAgent.create(
			agent=OurTensorForceAgent._SPECIFICATION_KEY,
			states={
				'type': 'float', 'shape': (self._m + self._n,), 'min_value': 0.0,
				'max_value': self._breach_level + self._delta_t},
			actions={'type': 'int', 'shape': (self._m + self._n,), 'num_values': OurTensorForceAgent._NUM_ACTIONS},
			memory=OurTensorForceAgent._MEMORY,
			update={'unit': 'timesteps', 'batch_size': OurTensorForceAgent._BATCH_SIZE},
			optimizer={
				'type': OurTensorForceAgent._OPTIMIZER_NAME,
				'learning_rate': learning_rate,
				'clipnorm':
					OurTensorForceAgent._OPTIMIZER_GRADIENT_CLIP_THRESHOLD if use_gradient_clipping else None},
			policy=self._model,
			objective=OurTensorForceAgent._OBJECTIVE,
			exploration=OurTensorForceAgent._EXPLORATION_RATE,
			reward_estimation={'horizon': OurTensorForceAgent._REWARD_HORIZON},
			saver=None if not OurTensorForceAgent._SAVE else {
				'directory': self._save_path, 'file_name': OurTensorForceAgent._SAVE_NAME,
				'frequency': OurTensorForceAgent._SAVING_FREQUENCY})

	def act(self, states: Any) -> Any:
		return self._tensor_force_agent.act(states=states).tolist()

	def observe(self, reward: float, terminal: bool) -> int:
		return self._tensor_force_agent.observe(reward, terminal)

	def save(self, path: str, identifier: str) -> None:
		self._tensor_force_agent.save(
			directory=path,
			filename='-'.join((OurTensorForceAgent.NAME, identifier)),
			format='numpy',
			append='episodes')


class OurProximalPolicyAgent(Agent):

	NAME: str = 'ppo-agent'

	_SPECIFICATION_KEY: str = 'ppo'
	_NUM_ACTIONS: int = 2
	_BATCH_SIZE: int = 64
	_LEARNING_RATE: float = 1e-3

	_SAVE: bool = False
	_SAVE_NAME: str = NAME
	_SAVING_FREQUENCY: int = 5 * 60

	def __init__(
			self,
			m: int,
			n: int,
			breach_level: float,
			delta_t: float,
			learning_rate: float,
			timeout_time: int,
			save_path: str,
			model: TensorForceModel) -> None:
		super(OurProximalPolicyAgent, self).__init__(m, n)
		self._breach_level = breach_level
		self._delta_t = delta_t
		self.timeout_time = timeout_time
		self._save_path = save_path
		self._model = model
		self._ppo_agent: TensorForceAgent = TensorForceAgent.create(
			agent=OurProximalPolicyAgent._SPECIFICATION_KEY,
			states={
				'type': 'float', 'shape': (self._m + self._n,), 'min_value': 0.0,
				'max_value': self._breach_level + self._delta_t},
			actions={'type': 'int', 'shape': (self._m + self._n,), 'num_values': OurProximalPolicyAgent._NUM_ACTIONS},
			max_episode_timesteps=self.timeout_time,
			batch_size=OurProximalPolicyAgent._BATCH_SIZE,
			learning_rate=learning_rate,
			network=self._model,
			saver=None if not OurProximalPolicyAgent._SAVE else {
				'directory': self._save_path, 'filename': OurProximalPolicyAgent._SAVE_NAME,
				'frequency': OurProximalPolicyAgent._SAVING_FREQUENCY})

	def act(self, states: Any) -> List[Action]:
		out = self._ppo_agent.act(states=states)
		return out.tolist()

	def observe(self, reward: float, terminal: bool) -> int:
		return self._ppo_agent.observe(reward, terminal)

	def save(self, path: str, identifier: str) -> None:
		self._ppo_agent.save(
			directory=path,
			filename='-'.join((OurProximalPolicyAgent.NAME, identifier)),
			format='numpy',
			append='episodes')


class NonAgent(Agent):

	_NUM_PERFORMED_UPDATES: int = 0  # this agent never updates itself

	def __init__(self, m: int, n: int, repair_threshold: int) -> None:
		super(NonAgent, self).__init__(m, n)
		self._repair_threshold = repair_threshold

	def act(self, states: Any) -> List[Action]:
		actions: List[int] = list()
		for i, x_t in enumerate(states):
			if x_t >= self._repair_threshold:
				actions.append(1)
			else:
				actions.append(0)
		return actions

	def observe(self, reward: float, terminal: bool) -> int:
		return NonAgent._NUM_PERFORMED_UPDATES

	def save(self, path: str, identifier: str) -> None:
		pass  # silently skip saving


class RandomAgent(Agent):

	_NUM_PERFORMED_UPDATES: int = 0  # this agent never updates itself

	def __init__(self, m: int, n: int) -> None:
		super(RandomAgent, self).__init__(m, n)
		self._action_interval: int = 0
		self.random_generator: Random = Random()
		self.random_generator.seed(999999)

	def act(self, states: Any) -> List[Action]:
		actions = list()
		for _ in range(0, (self._m + self._n)):
			action = self.random_generator.randint(0, 1)
			actions.append(action)
		return actions

	def observe(self, reward: float, terminal: bool) -> int:
		return RandomAgent._NUM_PERFORMED_UPDATES

	def save(self, path: str, identifier: str) -> None:
		pass  # silently skip saving
