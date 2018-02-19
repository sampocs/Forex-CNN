import tensorflow as tf
import numpy as np 
from random import randint
from gen_train_set import *
import time
import json

mu = raw_mu
var = raw_var

#Load data sets
train, val, test = gen_raw_class(one_hot=True, cnn_shape=True)
x_train, y_train = train
x_val, y_val = val 
x_test, y_test = test
x_dim = (x_train.shape[1], x_train.shape[2])

#Config
batch_size = 16
num_classes = 3
num_fc = 128
flat_nodes = 3200
learning_rate = 0.00085
epoch_len = 3000
num_epochs = 20

def get_batch(data, batch_size=batch_size):
	x, y = data
	batch_x = np.empty((batch_size, x.shape[1], x.shape[2], 1))
	batch_y = np.empty((batch_size, num_classes))
	for i in range(batch_size):
		r = np.random.randint(x.shape[0])
		batch_x[i], batch_y[i] = x[r], y[r]

	return batch_x, batch_y

def conv_weights(shape, name):
	#Xavier initilization
	W = tf.random_uniform(shape) * tf.sqrt(2.0/(shape[0]**2))
	return tf.Variable(W, name=name)

def fc_weights(shape, name):
	#Xavier initilization
	W = tf.random_uniform(shape) * tf.sqrt(2.0/shape[0])
	return tf.Variable(W, name=name)

def bias(shape, name):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(x, size):
	return tf.nn.max_pool(x, ksize=[1, size, 1, 1], strides=[1, size, 1, 1], padding='SAME')

#INPUT/LABEL
#X-dim: Batch, height, width, depth
x = tf.placeholder(tf.float32, shape=[None, x_dim[0], x_dim[1], 1], name='x')
#Y-dim: batch, one-hot class (sell, hold, buy) 
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

#WEIGHTS/BIAS
W_conv1 = conv_weights([4, 40, 1, 16], 'W_conv1')
W_conv2 = conv_weights([4, 1, 16, 16], 'W_conv2')
W_conv3 = conv_weights([3, 1, 16, 32], 'W_conv3')
W_conv4 = conv_weights([3, 1, 32, 32], 'W_conv4')

b_conv1 = bias([16], 'b_conv1')
b_conv2 = bias([16], 'b_conv2')
b_conv3 = bias([32], 'b_conv3')
b_conv4 = bias([32], 'b_conv4')

W_fc1 = fc_weights([flat_nodes, num_fc], 'W_fc1')
W_fc2 = fc_weights([num_fc, 3], 'W_fc2') 
b_fc1 = bias([num_fc], 'b_fc1')
b_fc2 = bias([3], 'b_fc2')


#BUILD
#Layer 1: Convolution -> batch normalization -> relu 
l1_conv = conv2d(x, W_conv1) + b_conv1
l1_bn = tf.layers.batch_normalization(l1_conv)
l1_prime = tf.nn.relu(l1_bn)

#Layer 2: Convolution -> batch normalization -> relu -> max pooling
l2_conv = conv2d(l1_prime, W_conv2) + b_conv2
l2_bn = tf.layers.batch_normalization(l2_conv)
l2_prime = tf.nn.relu(l2_bn)
l2_pool = max_pool(l2_prime, 2)

#Layer 3: Convolution -> batch normalization -> relu
l3_conv = conv2d(l2_pool, W_conv3) + b_conv3
l3_bn = tf.layers.batch_normalization(l3_conv)
l3_prime = tf.nn.relu(l3_bn)

#Layer 4: Convolution -> batch normalization -> relu -> max pooling
l4_conv = conv2d(l3_prime, W_conv4) + b_conv4
l4_bn = tf.layers.batch_normalization(l4_conv)
l4_prime = tf.nn.relu(l4_bn)
l4_pool = max_pool(l4_prime, 2)

#Layer 5: Flatten -> fully connected -> relu -> fully connected
l5_flat = tf.reshape(l4_pool, [-1, flat_nodes])
l5_fc = tf.matmul(l5_flat, W_fc1) + b_fc1
l5_prime = tf.nn.relu(l5_fc)
y = tf.add(tf.matmul(l5_prime, W_fc2), b_fc2, name="y")

#Loss: Softmax -> cross entropy loss -> average -> ADAM Optimizer
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

#Accuracy
with tf.name_scope("metrics"):
	correct_predicition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
	#precision = tf.metrics.precision(y_, y)
	#recall = tf.metrics.recall(y_, y)

#Tensorboard 
#Store histograms
tf.summary.histogram("W_conv1", W_conv1)
tf.summary.histogram("W_conv2", W_conv2)
tf.summary.histogram("W_conv3", W_conv3)
tf.summary.histogram("W_conv4", W_conv4)
tf.summary.histogram("fc_1", W_fc1)
tf.summary.histogram("fc_2", W_fc2)

#Loss/Accuracy/Precision/Recall
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)
#tf.summary.scalar("Precision", precision)
#tf.summary.scalar("Recall", recall)

with tf.Session() as sess:
	train_writer = tf.summary.FileWriter("/tmp/forex_cnn/train", sess.graph)
	val_writer = tf.summary.FileWriter("/tmp/forex_cnn/val")
	merged = tf.summary.merge_all()

	sess.run(tf.global_variables_initializer())

	for i in range(1, num_epochs + 1):
		for j in range(1, epoch_len + 1):

			batch_x, batch_y = get_batch(train)

				#Report loss at every 1000th iteration
			if j % 1000 == 0 and j != epoch_len:
				batch_loss, _ = sess.run([loss, train_step],
												feed_dict={ x: batch_x, y_: batch_y })
				print "Iteration {}/{} for Epoch {}, Loss: {}".format(j, epoch_len, i, batch_loss)
				
			#Report batch accuracy after each epoch
			elif j == epoch_len:
				x_train_batch, y_train_batch = get_batch(train, 50000)
				summary, train_acc, train_loss = sess.run([merged, accuracy, loss], feed_dict= { x: x_train_batch, y_: y_train_batch })
				train_writer.add_summary(summary, i)

				summary, val_acc = sess.run([merged, accuracy], feed_dict={ x: x_val, y_: y_val })
				val_writer.add_summary(summary, i)

				print "Epoch: {}, Loss: {}, Train Acc: {}, Val Acc: {}".format(i, train_loss, train_acc, val_acc)
					
			else:
				_ = sess.run([train_step], feed_dict={ x: batch_x, y_: batch_y })

#tensorboard --logdir /tmp/forex_cnn/
