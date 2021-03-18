import numpy as np
import tensorflow as tf
import pandas as pd
import socket
import time
from os import mkdir

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
from ppo_agent import PPO_Agent


if __name__ == '__main__':
	# set params
	tf.random.set_seed(999999)
	start_time = time.time()

	dyke_1_m = 1
	dyke_2_n = 1
	alpha = 5
	beta = 0.2
	"""
	properties gamma increment
	mean = alpha * beta
	variance = alpha * beta^2
	"""
	c_pm = 1
	c_cm = 3
	c_f = 0
	c_s = 100

	delta_t = 0.05
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
	max_episode_time_steps = int(1e4)

	# train an auto-encoder to supply later
	dyke_enc: Model = dyke_auto_encoder(
		dyke_env_params=env_params,
		num_samples=max_episode_time_steps,
		encoding_size=5)

	# set the agent
	machine_name = socket.gethostname()
	path = ""
	if machine_name[0:7] == "pg-node":
		path = "/data/s3178471/"
	dir_agent = "agent"
	try:
		if path == "":
			mkdir(dir_agent)
		else:
			mkdir(path + dir_agent)
	except FileExistsError:
		print("agent directory already exists")

	# load agent
	"""
	most simple agent, besides monte-carlo estimation  
	"""
	agent = Tensorforce_Agent(dyke_1_m=dyke_1_m,
					  dyke_2_n=dyke_2_n,
					  delta_t=delta_t,
					  L=L,
					  path=(path + dir_agent))
	"""
	agent = PPO_Agent(dyke_1_m=dyke_1_m,
					  dyke_2_n=dyke_2_n,
					  max_episode_timesteps=max_episode_time_steps,
					  delta_t=delta_t,
					  L=L,
					  path=(path + dir_agent))
	"""
	# set params
	t = 0
	save_time = 10
	print_info = 1000
	statuses = []
	terminal = False

	states_data = []
	reward_list = []
	mean_reward_list = []

	print(f"training {agent.agent_name} starts")
	# sample from the environment by doing nothing continuously
	for j in range(0, max_episode_time_steps):
		state = env.observe_state()
		actions = agent.agent.act(states=np.array(state)).tolist()  # tensorforce agent
		successful = env.take_action(actions=actions)

		state_after_action = env.observe_state()
		reward = env.get_reward()
		agent.agent.observe(terminal=terminal, reward=np.array(reward))
		t += delta_t
		reward_list.append(reward)

		# storing the time, reward, actions and dyke states
		# states_data.append([t] + [reward] + state_after_action + actions)
		if j == save_time:
			mean_reward = sum(reward_list) / len(reward_list)
			states_data.append([j] + [mean_reward])

			save_time += 10
			reward_list = []
			mean_reward_list.append(mean_reward)

			if j == print_info:
				print(f"current time {j}")
				print(f"mean reward {sum(mean_reward_list) / len(mean_reward_list)}")
				print_info += 1000
				mean_reward_list = []

	# save agent
	print("training done, saving agent")
	agent.agent.save(directory=(path + dir_agent), format='numpy', append='episodes')
	print("agent saved, saving database")

	# preparing dyke states into csv
	states_data = pd.DataFrame(states_data)
	states_data.columns = ["time"] + ["mean_reward"]
	#states_data.columns = ["time"] + ["reward"]  +\
	#					  [f"dyke_1_{i}" for i in range(1, (dyke_1_m+1))] + [f"dyke_2_{i}" for i in range(1, (dyke_2_n+1))] +\
	#					  [f"action_dyke_1_{i}" for i in range(1, (dyke_1_m + 1))] + [f"action_dyke_2_{i}" for i in range(1, (dyke_2_n + 1))]

	file_name = f"states_data.csv"
	states_data.to_csv((path + file_name), index=False)
	csv_info = open("csv_info.txt", "w+")
	csv_info.write(str(env_params))
	csv_info.close()
	print("all files successfully saved")

	# plot the state of the environment over time
	out: Tuple[Figure, Axes] = plt.subplots()
	plt.step(x=states_data.loc[:, "time"], y=states_data.loc[:, "mean_reward"].rolling(window=500).mean(), color='k')
	plt.xlabel('Time')
	plt.ylabel('Reward')
	plt.title('Deterioration Levels over Time')
	plt.show(block=True)

	# provide essential experimental information
	t_time = (time.time() - start_time)
	t_hours = t_time // 60 // 60
	t_min = (t_time - (t_hours * 60 * 60)) // 60
	t_seconds = (t_time - (t_min * 60) - (t_hours * 60 * 60))

	print(f"\nThe total run time"
		  f"\n\tHours:      {t_hours}"
		  f"\n\tMinutes:    {t_min}"
		  f"\n\tSeconds:    {round(t_seconds, 2)}")



"""
	# # plot the state of the environment over time
	#out: Tuple[Figure, Axes] = plt.subplots()
	#plt.step(x=states_data.loc[:, "time"], y=states_data.loc[:, "reward"].rolling(window=2500).mean(), color='k')
	#plt.xlabel('Time')
	#plt.ylabel('Reward')
	#plt.title('Deterioration Levels over Time')
	#plt.show(block=True)
"""