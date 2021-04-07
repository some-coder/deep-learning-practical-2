import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple, Type

from agent import Agent, EpsilonGreedyAgent, UpperConfidenceBoundAgent, GradientBanditAgent
from environment import Environment, GaussBanditEnvironment, StatelessDykeEnvironment, DiscreteDykeEnvironment
from dynamic_programming import PolicyIterator

from matplotlib.figure import Figure
from matplotlib.axes import Axes


REPEAT_LOG_INTERVAL: int = int(1e1)


def run(
		e: Type[Environment],
		e_params: Dict[str, Any],
		a: Type[Agent],
		a_params: Dict[str, Any],
		reset_agent: bool,
		timeout: int,
		repeats: int = 1) -> np.ndarray:
	# set up the environment and agent
	env = e(**e_params)
	a_params.update({'states': env.possible_states})
	a_params.update({'actions': env.possible_actions})
	agn = a(**a_params)
	# perform the main loop
	rewards: np.ndarray = np.zeros(shape=(repeats, timeout))
	for repeat in range(repeats):
		if (repeat + 1) % REPEAT_LOG_INTERVAL == 0:
			print('Repetition %d/%d.' % (repeat + 1, repeats))
		t: int = 0
		state = env.start()
		action = agn.decision(state)
		while t < timeout and not env.has_stopped():
			reward, state = env.proceed(action)
			agn.experience(reward)
			action = agn.decision(state)
			rewards[repeat, t] = reward
			t += 1
		env.reset()
		if reset_agent:
			agn.reset()
	return rewards


def demo(
		env: Type[Environment],
		env_params: Dict[str, Any],
		agent_param_map: Dict[Type[Agent], Dict[str, Any]],
		steps: int,
		repeats: int,
		reset_agent: bool,
		plot_options: Dict[str, Any]) -> None:
	# train the agents
	ys: np.ndarray = np.zeros(shape=(len(agent_param_map), steps))
	for index in range(len(agent_param_map)):
		a: Type[Agent] = list(agent_param_map.keys())[index]
		a_params: Dict[str, Any] = agent_param_map[a]
		ys[index, :] = run(
			env, env_params, a, a_params, reset_agent=reset_agent, timeout=steps,
			repeats=repeats).mean(axis=0)
	# make the plot
	sp: Tuple[Figure, Axes] = plt.subplots()
	x: List[int] = list(range(steps))
	colors: Tuple[str, str, str] = plot_options['colors']
	labels: Tuple[str, str, str] = plot_options['labels']
	for index in range(len(agent_param_map)):
		sp[1].plot(x, ys[index, :], colors[index] + '-', label=labels[index])
	sp[1].set_title(plot_options['title'])
	sp[1].legend(loc=plot_options['legend_loc'])
	sp[0].tight_layout()
	plt.show(block=True)


def bandits_demo(
		steps: int = int(1e3),
		env_params: Optional[Dict[str, Any]] = None,
		repeats: int = int(2e3)) -> None:
	if env_params is None:
		env_params = {'n': 10, 'mu': 0.0, 'std': 1.0, 'noise': 1.0}
	agent_param_map: Dict[Type[Agent], Dict[str, Any]] = \
		{
			EpsilonGreedyAgent: {'epsilon': 0.1, 'alpha': 0.1, 'optimism': 5.0},
			UpperConfidenceBoundAgent: {'c': 2.0},
			GradientBanditAgent: {'alpha': 0.1}
		}
	plot_opts: Dict[str, Any] = \
		{
			'colors': ('r', 'g', 'b'),
			'labels': ('Epsilon-greedy', 'UCB', 'Gradient bandit'),
			'title': 'N-armed bandits task',
			'legend_loc': 'lower right'
		}
	demo(GaussBanditEnvironment, env_params, agent_param_map, steps, repeats, True, plot_opts)


def stateless_dyke_demo(
		steps: int = int(1e3),
		env_params: Optional[Dict[str, Any]] = None,
		repeats: int = int(1e2)):
	if env_params is None:
		env_params = {
			'start_level': 0.0, 'breach_level': 2e-2,
			'reward_supplier': StatelessDykeEnvironment.StandardStatelessDykeRewardSupplier(1e-2, 0e0, 1e0, 1e1),
			'gamma_scale': 0.4, 'gamma_shape': 0.5}
	agent_param_map: Dict[Type[Agent], Dict[str, Any]] = \
		{
			EpsilonGreedyAgent: {'epsilon': 0.0, 'alpha': 0.1, 'optimism': 4.0},
			UpperConfidenceBoundAgent: {'c': 2.0},
			GradientBanditAgent: {'alpha': 0.1}
		}
	plot_opts: Dict[str, Any] = \
		{
			'colors': ('r', 'g', 'b'),
			'labels': ('Epsilon-greedy', 'UCB', 'Gradient bandit'),
			'title': 'Stateless dyke task',
			'legend_loc': 'lower right'
		}
	demo(StatelessDykeEnvironment, env_params, agent_param_map, steps, repeats, True, plot_opts)


if __name__ == '__main__':
	dde = DiscreteDykeEnvironment(
		dyke_lengths=(2, 2), k=5, breach_level=1.0,
		reward_supplier=DiscreteDykeEnvironment.StandardDiscreteDykeRewardSupplier(
			fixed=1e-2, preventive=1e-1, corrective=1e0, societal=1e2),
		gamma_scale=0.4, gamma_shape=0.5)
	pi = PolicyIterator(env=dde, discount=1e-1)
	pi.iterate()
	print('FINAL POLICY')
	for s in dde.possible_states:
		print('\t[s=%s] a=%d.' % (str(s), pi.policy[s]))
