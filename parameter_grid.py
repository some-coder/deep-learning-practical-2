import itertools as it

from agent import Agent, NonAgent, OurProximalPolicyAgent, AgentParameterMap, AgentParameterMapPair
from enum import Enum
from environment import Environment
from typing import cast, Set, Tuple, Type

from tensorflow.keras import Model


class NetworkSpecification:

	class NetworkType(Enum):
		FEED_FORWARD = 'feed-forward'
		RECURRENT = 'recurrent'

	def __init__(self, net_type: NetworkType, num_layers: int, max_pooling: bool) -> None:
		self._net_type = net_type
		self._num_layers = num_layers
		self._max_pooling = max_pooling

	def build(self) -> Model:
		raise NotImplementedError

	def __str__(self) -> str:
		return 'NetworkSpec(type=\'%s\', num_layers=%d, max_pooling=%s)' % \
			(self._net_type.value, self._num_layers, str(self._max_pooling))


class AgentSpecification:

	def __init__(self, agent: Type[Agent], params: AgentParameterMap) -> None:
		self._agent = agent
		self._params = params

	def build(self) -> Agent:
		return self._agent(**self._params)

	def __str__(self) -> str:
		return 'AgentSpec(agent=\'%s\', params=%s)' % (str(self._agent), str(self._params))


class Configuration:

	def __init__(
			self,
			reward_fn: Environment.RewardFunction,
			learning_rate: float,
			gradient_clipping: bool,
			agent_spec: AgentSpecification,
			network_spec: NetworkSpecification) -> None:
		self._reward_fn = reward_fn
		self._learning_rate = learning_rate
		self._gradient_clipping = gradient_clipping
		self._agent_spec = agent_spec
		self._network_spec = network_spec

	def __str__(self) -> str:
		return 'Configuration(rew_fn=%s, lr=%.3lf, g_clip=%s, agent_spec=%s, net_spec=%s)' % \
			(
				self._reward_fn.value, self._learning_rate, str(self._gradient_clipping), str(self._agent_spec),
				str(self._network_spec)
			)


def network_specification_grid(
		network_types: Set[NetworkSpecification.NetworkType],
		layer_numbers: Set[int],
		max_pooling_options: Set[bool]) -> Set[NetworkSpecification]:
	return cast(Set[NetworkSpecification], set(it.product(network_types, layer_numbers, max_pooling_options)))


def configuration_grid(
		reward_functions: Set[Environment.RewardFunction],
		learning_rates: Set[float],
		gradient_clipping_options: Set[bool],
		apm_pairs: Tuple[AgentParameterMapPair, ...],
		net_specs: Set[NetworkSpecification]) -> Tuple[Configuration, ...]:
	return cast(
		Tuple[Configuration, ...],
		tuple(it.product(reward_functions, learning_rates, gradient_clipping_options, apm_pairs, net_specs)))


if __name__ == '__main__':
	grid = configuration_grid(
		reward_functions={Environment.RewardFunction.STANDARD},
		learning_rates={1e-2, 1e-3},
		gradient_clipping_options={True},
		# Note: we do not have to specify all parameters for agents; we can fill those in for all agents later.
		apm_pairs=(
			(NonAgent, {'m': 5, 'n': 3, 'maintenance_interval': 5}),
			(OurProximalPolicyAgent, {
				'm': 5, 'n': 3, 'breach_level': 1e0, 'delta_t': 1e-2, 'timeout_time': int(1e2)})),
		net_specs=network_specification_grid(
			network_types={NetworkSpecification.NetworkType.FEED_FORWARD},
			layer_numbers={2, 8},
			max_pooling_options={False}))
