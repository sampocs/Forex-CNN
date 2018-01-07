import tensorflow as tf
import numpy as np 
from random import randint
from gen_train_set import *

mu = raw_mu
var = raw_var
train, val, test = gen_raw_class()

batch_size = 16

def get_batch(data):
	examp = np.empty((batch_size, prior, data[0][0].shape[1]))
	label = np.empty((batch_size))
	for i in range(batch_size):
		instance = data[np.random.randint(data.shape[0])]
		examp[i], label[i] = instance[0], instance[1]

	return examp, label 


