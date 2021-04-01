import copy as cp
import itertools as it

from agent import Agent, AgentParameterMap, AgentParameterMapPair, NonAgent, OurTensorForceAgent, \
	OurProximalPolicyAgent, RandomAgent, TensorForceModel
from enum import Enum
from environment import Environment
from typing import cast, Any, Dict, List, Set, Tuple, Type


class NetworkSpecification:

	_LAYER_MINIMUM: int = 2
	_DENSE_NUM_NODES: int = 10
	_ACTIVATION: str = 'relu'
	_LSTM_NUM_NODES: int = 10
	_LSTM_HORIZON: int = int(1e1)  # Steps of temporal back-propagation. Makes back-propagation through time feasible.

	class NetworkType(Enum):
		FEED_FORWARD = 'feed-forward'
		RECURRENT = 'recurrent'

	def __init__(self, net_type: NetworkType, num_layers: int, max_pooling: bool) -> None:
		self._net_type = net_type
		if num_layers < NetworkSpecification._LAYER_MINIMUM:
			print('You need to specify at least %d layers (got %d).' % (NetworkSpecification._LAYER_MINIMUM, num_layers))
		self._num_layers = num_layers
		self._max_pooling = max_pooling

	def build(self) -> TensorForceModel:
		layers: List[Dict[str, Any]] = []
		mdl: TensorForceModel = {'network': layers}
		if self._max_pooling:
			layers.append({'type': 'pooling', 'reduction': 'max'})
		if self._net_type == NetworkSpecification.NetworkType.FEED_FORWARD:
			layers.append({
				'type': 'dense', 'size': NetworkSpecification._DENSE_NUM_NODES,
				'activation': NetworkSpecification._ACTIVATION})
		for _ in range(self._num_layers - (NetworkSpecification._LAYER_MINIMUM - (0 if self._max_pooling else 1))):
			layers.append({
				'type': 'dense', 'size': NetworkSpecification._DENSE_NUM_NODES,
				'activation': NetworkSpecification._ACTIVATION})
		if self._net_type == NetworkSpecification.NetworkType.RECURRENT:
			layers.append({
				'type': 'rnn', 'cell': 'lstm', 'size': NetworkSpecification._LSTM_NUM_NODES,
				'activation': NetworkSpecification._ACTIVATION, 'horizon': NetworkSpecification._LSTM_HORIZON})
		return mdl

	def __str__(self) -> str:
		return 'NetworkSpec(type=\'%s\', num_layers=%d, max_pooling=%s)' % \
			(self._net_type.value, self._num_layers, str(self._max_pooling))


NetworkParameterization = Tuple[NetworkSpecification.NetworkType, int, bool]


class AgentSpecification:

	_REQUIRE_ENVIRONMENT_INFO: Tuple[Type[Agent], ...] = (OurTensorForceAgent, OurProximalPolicyAgent)
	_REQUIRE_LEARNING_RATE: Tuple[Type[Agent], ...] = (OurTensorForceAgent, OurProximalPolicyAgent)
	_REQUIRE_GRADIENT_CLIPPING: Tuple[Type[Agent], ...] = (OurTensorForceAgent,)
	_REQUIRE_MODEL: Tuple[Type[Agent], ...] = (OurTensorForceAgent, OurProximalPolicyAgent)

	def __init__(self, agent: Type[Agent], params: AgentParameterMap) -> None:
		self._type = agent
		self.params = params

	def complete_specification(
			self,
			env_spec: Dict[str, Any],
			learning_rate: float,
			gradient_clipping: bool,
			model: NetworkSpecification,) -> None:
		d: Dict[str, Any] = {'m': env_spec['m'], 'n': env_spec['n']}
		# add environment information
		keys: Tuple[str, ...] = tuple()
		if self._type in AgentSpecification._REQUIRE_ENVIRONMENT_INFO:
			keys += 'breach_level', 'delta_t'
		for key in keys:
			d[key] = env_spec[key]
		self.params.update(d)
		if self._type in AgentSpecification._REQUIRE_LEARNING_RATE:
			self.params.update({'learning_rate': learning_rate})
		# set gradient clipping
		if self._type in AgentSpecification._REQUIRE_GRADIENT_CLIPPING:
			self.params.update({'use_gradient_clipping': gradient_clipping})
		# add the network
		if self._type == AgentSpecification._REQUIRE_MODEL[0]:
			self.params.update({'model': model.build()})
		elif self._type == AgentSpecification._REQUIRE_MODEL[1]:
			mdl: TensorForceModel = model.build()
			self.params.update({'model': {'type': 'layered', 'layers': mdl['network']}})

	def build(self) -> Agent:
		return self._type(**self.params)

	def __str__(self) -> str:
		return 'AgentSpec(type=\'%s\', params=%s)' % (str(self._type), str(self.params))


ConfigurationParameterization = \
	Tuple[Environment.RewardFunction, float, bool, AgentParameterMapPair, NetworkSpecification]


class Configuration:

	def __init__(
			self,
			reward_fn: Environment.RewardFunction,
			learning_rate: float,
			gradient_clipping: bool,
			agent_spec: AgentSpecification,
			network_spec: NetworkSpecification) -> None:
		"""
		Constructs a specification for an Environment-Agent combination.

		Only a subset of the parameters of an ``agent.Agent`` need to be supplied. They are:

		* ``save_path`` for ``agent.OurTensorForceAgent``,
		* ``timeout_time`` and ``save_path`` for ``agent.OurProximalPolicyAgent``,
		* ``repair_threshold`` for ``agent.NonAgent``.
		* No parameters need to be specified for ``agent.RandomAgent``.

		:param reward_fn: The reward function to use. Options are listed in
			<tt>environment.Environment.RewardFunction</tt>.
		:param learning_rate: The learning rate the agent should use.
		:param gradient_clipping: Whether to let the agent use gradient clipping.
		:param agent_spec: The agent and their specific parameters. See detailed comments above.
		:param network_spec: The specification for the agent's network. Ignored for <tt>agent.NonAgent</tt>.
		"""
		self._reward_fn = reward_fn
		self._learning_rate = learning_rate
		self._gradient_clipping = gradient_clipping
		self._agent_spec = agent_spec  # will need further information from an environment
		self._network_spec = network_spec

	def build(self, env_spec: Dict[str, Any]) -> Tuple[Environment, Agent]:
		"""
		Instantiates one Environment-Agent pair.

		Supply all parameters to the `environment.Environment`, except the following, as these are already available
		to this `Configuration` instance:

		* `reward_fn`.

		:param env_spec: Parameters to supply to <tt>environment.Environment</tt>. See detailed comments above.
		:return: A two-tuple of <tt>environment.Environment</tt> and <tt>agent.Agent</tt>, respectively.
		"""
		env_dict: Dict[str, Any] = cp.deepcopy(env_spec)
		env_dict.update({'reward_fn': self._reward_fn})
		env_agent_spec = cp.deepcopy(self._agent_spec)
		env_agent_spec.complete_specification(
			env_spec, self._learning_rate, self._gradient_clipping, self._network_spec)
		return Environment(**env_dict), env_agent_spec.build()

	def __str__(self) -> str:
		return 'Configuration(rew_fn=%s, lr=%.3lf, g_clip=%s, agent_spec=%s, net_spec=%s)' % \
			(
				self._reward_fn.value, self._learning_rate, str(self._gradient_clipping), str(self._agent_spec),
				str(self._network_spec)
			)


def network_specification_grid(
		network_types: Set[NetworkSpecification.NetworkType],
		layer_numbers: Set[int],
		max_pooling_options: Set[bool]) -> Tuple[NetworkSpecification, ...]:
	nss = cast(
		Tuple[NetworkParameterization, ...],
		tuple(it.product(network_types, layer_numbers, max_pooling_options)))
	return tuple(NetworkSpecification(*net_par) for net_par in nss)


def _configurations(
		reward_functions: Set[Environment.RewardFunction],
		learning_rates: Set[float],
		gradient_clipping_options: Set[bool],
		apm_pair: AgentParameterMapPair,
		net_specs: Tuple[NetworkSpecification, ...]) -> Tuple[Configuration, ...]:
	"""
	Yields the configurations appropriate for the supplied agent-parametrisation pair.

	This method explicitly does not vary in axes that the agent is not receptive to.
	For instance, a ``agent.RandomAgent`` does nothing with a network model. This method
	thus does not provide said agent with multiple models, even though you may have
	specified multiple network specifications.

	:param reward_functions: The reward functions to vary with.
	:param learning_rates: The learning rates to vary with.
	:param gradient_clipping_options: The learning rate setting variations.
	:param apm_pair: A single agent-parametrisation pair.
	:param net_specs: The network specifications to vary with.
	:return: The configurations appropriate for this agent type.
	"""
	single_lr: Tuple[float] = (list(learning_rates)[0],)
	single_gc: Tuple[bool] = (list(gradient_clipping_options)[0],)
	single_ns: Tuple[NetworkSpecification] = (net_specs[0],)
	if apm_pair[0] in (OurTensorForceAgent,):
		# yield a full Cartesian product
		return tuple(it.product(reward_functions, learning_rates, gradient_clipping_options, (apm_pair,), net_specs))
	elif apm_pair[0] in (OurProximalPolicyAgent,):
		# drop variation in gradient clipping
		return tuple(it.product(reward_functions, learning_rates, single_gc, (apm_pair,), net_specs))
	elif apm_pair[0] in (NonAgent, RandomAgent):
		# only vary the reward function
		return tuple(it.product(reward_functions, single_lr, single_gc, (apm_pair,), single_ns))


def configuration_grid(
		reward_functions: Set[Environment.RewardFunction],
		learning_rates: Set[float],
		gradient_clipping_options: Set[bool],
		apm_pairs: Tuple[AgentParameterMapPair, ...],
		net_specs: Tuple[NetworkSpecification, ...]) -> Tuple[Configuration, ...]:
	cps: Tuple[ConfigurationParameterization, ...] = tuple()
	for apm_pair in apm_pairs:
		cps += _configurations(reward_functions, learning_rates, gradient_clipping_options, apm_pair, net_specs)
	out: Tuple[Configuration, ...] = tuple()
	for con_par in cps:
		ag_sp = AgentSpecification(con_par[3][0], con_par[3][1])
		out += (Configuration(con_par[0], con_par[1], con_par[2], ag_sp, con_par[4]),)
	return out
