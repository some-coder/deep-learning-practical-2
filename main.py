from typing import Any, Dict

from agent import NonAgent, OurProximalPolicyAgent, OurTensorForceAgent
from environment import Environment
from parameter_grid import configuration_grid, network_specification_grid, NetworkSpecification
from runner import Runner


if __name__ == '__main__':
	grid = configuration_grid(
		reward_functions={Environment.RewardFunction.STANDARD},
		learning_rates={1e-2, 1e-3},
		gradient_clipping_options={True},
		# Note: not all parameters need to be supplied. See
		apm_pairs=(
			(NonAgent, {'maintenance_interval': 5}),
			# (OurProximalPolicyAgent, {'timeout_time': int(1e2), 'save_path': '.'})),
			(OurTensorForceAgent, {'save_path': '.'})),
		net_specs=network_specification_grid(
			network_types={NetworkSpecification.NetworkType.FEED_FORWARD},
			layer_numbers={2, 8},
			max_pooling_options={False}))
	c = grid[3]
	print(c)
	env_spec: Dict[str, Any] = {
		'm': 5, 'n': 5, 'alpha': 4e-1, 'beta': 5e-1, 'c_pm': 1e-1, 'c_cm': 1e0, 'c_f': 1e-2, 'c_s': 1e2,
		'delta_t': 1e-2, 'breach_level': 1e0}
	en, ag = c.build(env_spec)
