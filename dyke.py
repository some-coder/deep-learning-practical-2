import numpy as np
import tensorflow as tf
import pandas as pd

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Any, Dict, Iterator, Tuple

from environment import Environment
from non_agent import Non_Agent
from tensorforce_agent import Tensorforce_Agent
from tpro_agent import TPRO_Agent


if __name__ == '__main__':
	tf.random.set_seed(999999)
	dyke_1_m = 20
	dyke_2_n = 20
	alpha = 5
	beta = 0.6
	"""
	properties gamma increment
	mean = alpha * beta * time
	std. deviation = alpha * beta^2 * time 
	"""
	c_pm = 1
	c_cm = 4
	c_f = 4
	c_s = 1000

	delta_t = 0.01
	L = 1

	# set up the environment
	env: Environment = Environment(
		m=dyke_1_m,
		n=dyke_2_n,
		alpha=alpha,
		beta=beta,
		c_pm=c_pm,
		c_cm=c_cm,
		c_f=c_f,
		c_s=c_s,
		delta_t=delta_t,
		L=L
	)
	# number of episodes
	max_episode_timesteps = 50000

	# set the agent
	#agent = Non_Agent(maintenance_interval=0.7, dyke_1_m=dyke_1_m, dyke_2_n=dyke_2_n)
	agent = Tensorforce_Agent(dyke_1_m=dyke_1_m, dyke_2_n=dyke_2_n)
	#agent = TPRO_Agent(dyke_1_m=dyke_1_m, dyke_2_n=dyke_2_n, max_episode_timesteps=max_episode_timesteps)

	# set params
	time = 0
	statuses = []
	terminal = False
	# sample from the environment by doing nothing continuously
	for j in range(0, max_episode_timesteps):
		state = env.observe_state()
		actions = agent.agent.act(states=np.array(state)).tolist()  # tensorforce agent
		#actions = agent.act(time=time) # fixed interval agent
		succesfull = env.take_action(actions=actions)
		reward = env.get_reward()
		agent.agent.observe(terminal=terminal, reward=np.array(-reward))
		#statuses.append([time, reward]) # collect reward
		statuses.append([time] + state) # collect states
		time += delta_t

	df = pd.DataFrame(statuses)
	#df.columns = ["time"] + ["reward"]
	df.columns = ["time"] + [f"dyke_1_{i}" for i in range(1, (dyke_1_m+1))] + [f"dyke_2_{i}" for i in range(1, (dyke_2_n+1))]

	# # plot the state of the environment over time
	out: Tuple[Figure, Axes] = plt.subplots()
	color: Iterator[np.array] = iter(plt.cm.get_cmap(name='rainbow')(X=np.linspace(start=0, stop=1, num=(dyke_1_m + dyke_1_m))))
	for key in range(0, (dyke_1_m + dyke_1_m)):
		c: np.array = next(color)  # 1-by-4 RGBA array
		#plt.step(x=df.loc[:,"time"], y=df.loc[:,"reward"], color=c)
		plt.step(x=df.loc[:, "time"], y=df.iloc[:, 1:-1], color=c)
	#out[1].set_ylim(top=c_s)
	#out[1].set_ylim(top=L+delta_t)
	plt.xlabel('Time')
	plt.ylabel('Deterioration level')
	#plt.ylabel('Reward')
	plt.title('Deterioration Levels over Time')
	plt.show(block=True)
