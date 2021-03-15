import numpy as np
import tensorflow as tf
import pandas as pd

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Any, Dict, Iterator, Tuple

from auto_encoder import dyke_auto_encoder
from environment import Environment
from non_agent import Non_Agent
from tensorflow.keras.models import Model
from tensorforce_agent import Tensorforce_Agent
from tpro_agent import TPRO_Agent


if __name__ == '__main__':
	tf.random.set_seed(999999)
	dyke_1_m = 10
	dyke_2_n = 10
	alpha = 5
	beta = 0.01 # small variance=0.0005
	# beta = 0.1 # medium variance=0.05
	# beta = 0.4472139 # large variance=1.0000
	"""
	properties gamma increment
	mean = alpha * beta * time
	variance = alpha * beta^2 * time 
	"""
	c_pm = 1
	c_cm = 1
	c_f = 0
	c_s = 0

	delta_t = 0.01
	L = 1

	# set up the environment
	env_params: Dict[str, Any] = \
		{
			'm': dyke_1_m,
			'n': dyke_2_n,
			'alpha': alpha,
			'beta': beta,
			'c_pm': c_pm,
			'c_cm': c_cm,
			'c_f': c_f,
			'c_s': c_s,
			'delta_t': delta_t,
			'L': L
		}
	env: Environment = Environment(**env_params)

	# number of episodes
	max_episode_time_steps = int(4e4)

	# train an auto-encoder to supply later
	dyke_enc: Model = dyke_auto_encoder(
		dyke_env_params=env_params,
		num_samples=max_episode_time_steps,
		encoding_size=5)

	# set the agent
	# agent = Non_Agent(maintenance_interval=0.7, dyke_1_m=dyke_1_m, dyke_2_n=dyke_2_n)
	agent = Tensorforce_Agent(dyke_1_m=dyke_1_m, dyke_2_n=dyke_2_n, auto_encoder=dyke_enc)
	# agent = TPRO_Agent(dyke_1_m=dyke_1_m, dyke_2_n=dyke_2_n, max_episode_timesteps=max_episode_timesteps)

	# set params
	time = 0
	statuses = []
	terminal = False

	states_data = []

	# sample from the environment by doing nothing continuously
	for j in range(0, max_episode_time_steps):
		state = env.observe_state()
		actions = agent.agent.act(states=np.array(state)).tolist()  # tensorforce agent
		#actions = agent.act(time=time) # fixed interval agent
		successful = env.take_action(actions=actions)

		state_after_action = env.observe_state()
		reward = env.get_reward()
		agent.agent.observe(terminal=terminal, reward=np.array(-reward))
		statuses.append([time, reward]) # collect reward
		#statuses.append([time] + state) # collect states
		time += delta_t

		# storing the time, reward, actions and dyke states
		states_data.append([time] + [reward] + state_after_action + actions)





	# preparing dyke states into csvs
	states_data = pd.DataFrame(states_data)
	states_data.columns = ["time"] + ["reward"]  +\
						  [f"dyke_1_{i}" for i in range(1, (dyke_1_m+1))] + [f"dyke_2_{i}" for i in range(1, (dyke_2_n+1))] +\
						  [f"action_dyke_1_{i}" for i in range(1, (dyke_1_m + 1))] + [f"action_dyke_2_{i}" for i in range(1, (dyke_2_n + 1))]

	states_data.to_csv(f"states_data.csv", index=False)
	csv_info = open("csv_info.txt", "w+")
	csv_info.write(str(env_params))
	csv_info.close()



	df = pd.DataFrame(statuses)
	df.columns = ["time"] + ["reward"]
	#df.columns = ["time"] + [f"dyke_1_{i}" for i in range(1, (dyke_1_m+1))] + [f"dyke_2_{i}" for i in range(1, (dyke_2_n+1))]

	# # plot the state of the environment over time
	out: Tuple[Figure, Axes] = plt.subplots()
	color: Iterator[np.array] = iter(plt.cm.get_cmap(name='rainbow')(X=np.linspace(start=0, stop=1, num=(dyke_1_m + dyke_1_m))))
	for key in range(0, (dyke_1_m + dyke_1_m)):
		c: np.array = next(color)  # 1-by-4 RGBA array
		plt.step(x=df.loc[:,"time"], y=df.loc[:,"reward"], color=c)
		#plt.step(x=df.loc[:, "time"], y=df.iloc[:, 1:-1], color=c)
	#out[1].set_ylim(top=c_s)
	#out[1].set_ylim(top=L+delta_t)
	plt.xlabel('Time')
	#plt.ylabel('Deterioration level')
	plt.ylabel('Reward')
	plt.title('Deterioration Levels over Time')
	plt.show(block=True)
