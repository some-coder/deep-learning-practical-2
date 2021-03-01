import numpy as np
import tensorflow as tf

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from model import DykeRepairEnv
from typing import Any, Dict, Iterator, Tuple

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment


if __name__ == '__main__':
	# set seed for reproducibility
	np.random.seed(1)

	# set parameters
	dk1: np.int32 = np.int32(3)
	dk2: np.int32 = np.int32(5)
	dkl: np.int32 = dk1 + dk2
	iter_length: int = int(1.75e2)
	thr: np.float64 = np.float64(1e0)

	# set up the environment
	py_train_env: py_environment.PyEnvironment = \
		DykeRepairEnv(
			len_dyke_1=dk1,
			len_dyke_2=dk2,
			prevent_cost=np.float64(0.5),
			correct_cost=np.float64(1.5),
			fixed_cost=np.float64(0.1),
			society_cost=np.float64(20.0),
			alpha=np.float64(5.0),
			beta=np.float64(0.2),
			threshold=thr)
	py_eval_env: py_environment.PyEnvironment = deepcopy(py_train_env)

	tf_train: tf_py_environment.TFPyEnvironment = \
		tf_py_environment.TFPyEnvironment(environment=py_train_env)

	# sample from the environment by doing nothing continuously
	# noop_action: np.array = np.array(dkl * [DykeRepairEnv.Action.NO_OPERATION.value], dtype=np.int32)
	# statuses: np.array = np.ndarray(shape=(0, dkl))
	# state: Dict[str, Any] = py_train_env.get_state()
	# statuses = np.vstack((statuses, state['state']))
	# for _ in range(iter_length):
	# 	time_step: ts.TimeStep = py_train_env.step(noop_action)
	# 	state = py_train_env.get_state()
	# 	statuses = np.vstack((statuses, state['state']))
	#
	# # plot the state of the environment over time
	# out: Tuple[Figure, Axes] = plt.subplots()
	# color: Iterator[np.array] = iter(plt.cm.get_cmap(name='rainbow')(X=np.linspace(start=0, stop=1, num=dkl)))
	# for key in range(0, dkl):
	# 	c: np.array = next(color)  # 1-by-4 RGBA array
	# 	plt.step(x=np.arange(0, statuses.shape[0]), y=statuses[:, key], color=c)
	# out[1].set_ylim(top=thr)
	# plt.xlabel('Time')
	# plt.ylabel('Deterioration level')
	# plt.title('Deterioration Levels over Time')
	# plt.show(block=True)
