from typing import Any, Dict

from agent import NonAgent, OurProximalPolicyAgent, OurTensorForceAgent
from environment import Environment
from parameter_grid import configuration_grid, network_specification_grid, NetworkSpecification
from runner import Runner


if __name__ == '__main__':
	timeout_time: int = int(1e4)  # Important: PPO as well as Runner need the *exact same* value!
	grid = configuration_grid(
		reward_functions={Environment.RewardFunction.STANDARD},
		learning_rates={1e-2, 1e-3},
		gradient_clipping_options={True},
		# Note: not all parameters need to be supplied. See ``parameter_grid.Configuration`` for details.
		apm_pairs=(
			(NonAgent, {'maintenance_interval': 5}),
			(OurTensorForceAgent, {'save_path': '.'}),
			(OurProximalPolicyAgent, {
				'timeout_time': timeout_time, 'save_path': '.'})),
		net_specs=network_specification_grid(
			network_types={NetworkSpecification.NetworkType.FEED_FORWARD},
			layer_numbers={2, 8},
			max_pooling_options={False}))
	env_spec: Dict[str, Any] = {
		'm': 5, 'n': 5, 'alpha': 4e-1, 'beta': 5e-1, 'c_pm': 1e-1, 'c_cm': 1e0, 'c_f': 1e-2, 'c_s': 1e2,
		'delta_t': 1e-2, 'breach_level': 1e0}
	experiment_tag: str = 'test-exp'
	for model_id, point in enumerate(grid):
		print('[MODEL %d/%d] %s.' % (model_id + 1, len(grid), str(point)))
		en, ag = point.build(env_spec)
		Runner.run(
			en, ag, identifier='-'.join((experiment_tag, str(model_id + 1))), save_agent=False, time_steps=timeout_time)
	Runner.save_experiment_identifiers(experiment_tag, configurations=grid, associated_ids=list(range(len(grid))))
