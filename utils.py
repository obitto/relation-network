import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def fc(inputs, output_shape, activation_fn=tf.nn.relu, name="fc"):
	"""
	fully connect layer
	"""
	output = slim.fully_connected(inputs, int(output_shape), activation_fn=activation_fn)
	return output

def conv2d(image, filters, kernel_size, name, strides = 2, activation = tf.nn.relu,
			padding = "same", is_train = True, batch_norm = True):
	"""
	convolution layer, use batch normalization if necessary.
	"""
	with tf.variable_scope(name) as scope:
		conv = tf.layers.conv2d(inputs = image, 
								filters = filters, 
								kernel_size=[kernel_size, kernel_size], 
								strides=(strides, strides),
								padding = "same", 
								activation= activation,
								name = name,trainable=is_train)
		if batch_norm == True:
			batch = tf.contrib.layers.batch_norm(conv, decay = 0.9, scale = True, trainable = is_train)
			return batch
		else:
			return conv

def tag_obj(o, i, d):
	"""
	tag the object with normalized coordinate
	"""
	coor = tf.tile(tf.expand_dims(
			[float(int(i / d)) / d * 2 - 1, (i % d) / d * 2 - 1], axis=0), [o.get_shape().as_list()[0], 1])
	o = tf.concat([o, tf.to_float(coor)], axis=1)
	return o
        
def g_theta( o_i, o_j, question, reuse = True, scope='g_theta'):
	"""
	combine the object pair and question embedding, then feed it through 4 MLP Layer.
	"""
	with tf.variable_scope(scope, reuse = reuse) as scope:
		g_1 = fc(tf.concat([o_i, o_j, question], axis=1), 256, name='g_1')
		g_2 = fc(g_1, 256, name='g_2')
		g_3 = fc(g_2, 256, name='g_3')
		g_4 = fc(g_3, 256, name='g_4')
		return g_4
		
class Config(object):
	"""
	Configuration for clevr model
	"""
	imageW = 80
	imageH = 80
	imageC = 3
	padW = 88
	padH = 88
	kernel_size = 3
	seq_length = 30
	answer_dim = 28
	lstm_dim = 128
	filters = 24
	vocab_size = 90
	embed_dim = 32
	batch_size = 64
	hidden_size =128
	max_iter = 140000
	save = 100
	learning_rate = 0.00025
	g_theta = [256, 256, 256, 256]
	image_shape = [batch_size,imageW, imageH,imageC]
	
class Config_SClevr(object):
	"""
	configuration for sort-of-clevr model
	"""
	imageW = 80
	imageH = 80
	imageC = 3
	padW = 80
	padH = 80
	kernel_size = 3
	answer_dim = 10
	question_dim = 11
	lstm_dim = 128
	filters = 24
	batch_size = 64
	hidden_size =128
	max_iter = 500000
	learning_rate = 0.0001
	save = 100
	image_shape = [batch_size,imageW, imageH,imageC]
