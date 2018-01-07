import numpy as np
from random import uniform
import time

prior = 100
k = 20
alpha = 0.005
oversample_thresh = 0.3
undersample_thresh = 0.3

#Load data
overfit = np.genfromtxt("data/overfit.csv", delimiter=',')
raw_train = np.genfromtxt("data/raw_train.csv", delimiter=',')
raw_val = np.genfromtxt("data/raw_val.csv", delimiter=',')
raw_test = np.genfromtxt("data/raw_test.csv", delimiter=',')

#Store mean and variance
raw_mu = np.sum(raw_train, axis=0) / (float(raw_train.shape[0]))
raw_var = np.sum(np.square(raw_train), axis=0) / (float(raw_train.shape[0]))

"""Input is 100 previous prices, label represents price movement (average of k points)"""
def gen_raw_class():
	#train, val, test
	sets = []
	indx = 0
	for set_type in [raw_train, raw_val, raw_test]:
	
		x = np.empty((set_type.shape[0] * 2, prior, set_type.shape[1]))
		y = np.empty((set_type.shape[0] * 2, 1))
		for i in range(set_type.shape[0] - prior - k):
			examp = np.copy(set_type[i : i + prior])

			#Normalize
			examp -= raw_mu
			examp /= (raw_var ** (1/2.0))

			l_index = i + prior
			#average of previous k close-mid prices 
			m_b = np.mean(set_type[l_index - k : l_index], axis=0)[0]
			#average of next k close-mid prices
			m_a = np.mean(set_type[l_index + 1: l_index + k + 1], axis=0)[0]

			#decrease = -1, increase = 1, stable = 0
			if m_b > m_a * (1 + alpha):
				label = -1
			elif m_b < m_a * (1 - alpha):
				label = 1 
			else:
				label = 0
			#label = np.array([label])

			#Balance dataset by oversampling from -1, 1 class
			if label != 0 and uniform(0, 1) < oversample_thresh:
				x[indx], y[indx] = examp, label
				indx += 1
			#And undersampling from 0 class
			elif label == 0 and uniform(0, 1) < undersample_thresh:
				continue
			x[indx], y[indx] = examp, label
			indx += 1

		group = np.array(zip(x[:indx], y[:indx]))
		np.random.shuffle(group)
		x, y = zip(*list(group))
		sets.append((x, y))

	return sets


"""Input is 100 previous prices, label is 101th closeMid price"""
def gen_raw_reg():
	#train, val, test
	sets = []
	for set_type in [raw_train, raw_val, raw_test]:

		x = np.empty((set_type.shape[0], prior, set_type.shape[1]))
		y = np.empty((set_type.shape[0], 1))
		for i in range(set_type.shape[0] - prior - 1):
			examp = np.copy(set_type[i : i + prior])

			#Normalize
			examp -= raw_mu
			examp /= (raw_var ** (1/2.0))

			label = set_type[i + prior][0]
			#label = np.array([label])

			x[i] = examp
			y[i] = label

		group = np.array(zip(x, y))
		np.random.shuffle(group)
		x, y = zip(*list(group))
		sets.append((x, y))

	return sets

