"""
Project: Deep Learning, experimental environment for Peregrine
Made By: Arno Kasper
Version: 1.0.0
"""
from typing import Any, Dict

from agent import NonAgent, OurProximalPolicyAgent, OurTensorForceAgent, RandomAgent
from environment import Environment
from parameter_grid import configuration_grid, network_specification_grid, NetworkSpecification
from runner import Runner

class Experiment_Manager(object):

    # Creat a batch of experiments with a upper an lower limit
    def __init__(self, lower, upper):
        """
        initialize experiments integers
        :param lower: lower boundary of the exp number
        :param upper: upper boundary of the exp number
        """
        self.lower = lower
        self.upper = upper
        self.count_experiment = 0
        self.exp_manager()

    def exp_manager(self):
        """
        define the experiment manager who controls the experiments
        :return: void
        """
        # use a loop to illiterate multiple experiments from the exp_dat list
        timeout_time: int = int(8e5)  # Important: PPO as well as Runner need the *exact same* value!
        grid = configuration_grid(
            reward_functions={Environment.RewardFunction.STANDARD},
            learning_rates={1e-2, 1e-3},
            gradient_clipping_options={True},
            # Note: not all parameters need to be supplied. See ``parameter_grid.Configuration`` for details.
            apm_pairs=(
                (RandomAgent, {}),
                (NonAgent, {'repair_threshold': 0.1}),
                (NonAgent, {'repair_threshold': 0.2}),
                (NonAgent, {'repair_threshold': 0.3}),
                (NonAgent, {'repair_threshold': 0.4}),
                (NonAgent, {'repair_threshold': 0.5}),
                (NonAgent, {'repair_threshold': 0.6}),
                (NonAgent, {'repair_threshold': 0.7}),
                (NonAgent, {'repair_threshold': 0.8}),
                (NonAgent, {'repair_threshold': 0.9}),
                (NonAgent, {'repair_threshold': 1.0}),
                (OurTensorForceAgent, {'save_path': '.'}),
                (OurProximalPolicyAgent, {
                    'timeout_time': timeout_time, 'save_path': '.'})),
            net_specs=network_specification_grid(
                network_types={NetworkSpecification.NetworkType.FEED_FORWARD},
                layer_numbers={2, 8},
                max_pooling_options={False}))
        env_spec: Dict[str, Any] = {
            'm': 5, 'n': 5, 'alpha': 5, 'beta': 0.2, 'c_pm': 1, 'c_cm': 10, 'c_f': 4, 'c_s': 250,
            'delta_t': 0.01, 'breach_level': 1}
        experiment_tag: str = 'full_exp'
        # start running experiments
        print(f"there are: {len(grid)} experiments")

        # quick fix for names
        model_names = [
            "RandomAgent",
            "NonAgent_01",
            "NonAgent_02",
            "NonAgent_03",
            "NonAgent_04",
            "NonAgent_05",
            "NonAgent_06",
            "NonAgent_07",
            "NonAgent_08",
            "NonAgent_09",
            "NonAgent_10",
            "OurTensorForceAgent_1",
            "OurTensorForceAgent_2",
            "OurTensorForceAgent_3",
            "OurTensorForceAgent_4",
            "OurProximalPolicyAgent_1",
            "OurProximalPolicyAgent_2",
            "OurProximalPolicyAgent_3",
            "OurProximalPolicyAgent_4",
        ]

        for model_id, point in enumerate(grid[self.lower: (self.upper + 1)]):
            print('[MODEL %d/%d] %s.' % (model_id + 1, len(grid), str(point)))
            env, agn = point.build(env_spec)
            Runner.run(
                env=env, agn=agn, identifier='_'.join((experiment_tag, str(model_names[self.lower + model_id]))), save_agent=False,
                time_steps=timeout_time)
        print("experiment done, saving files")
        Runner.save_experiment_identifiers(experiment_tag, configurations=grid, associated_ids=list(range(len(grid))))

if __name__ == '__main__':
    Experiment_Manager(0,0)