from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 

class Resnet(object):
	"""docstring for Resnet"""
	def __init__(self,is_training,keep_prob,stack_num=3,num_classes=10):
		super(Resnet, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
		self.initializer = tf.contrib.layers.xavier_initializer()
		self.stack_num = stack_num
		self.keep_prob = keep_prob

	def residual_block(self,inputs,output_channel,stride=[1,1]):
		residual = tf.identity(inputs)
		input_channel = residual.shape[-1]
		x_width = residual.shape[-2]
		inputs = self.conv2d(inputs,output_channel,stride=stride[0])
		inputs = self.conv2d(inputs,output_channel,stride=stride[1],relu=False)
		inputs_width = inputs.shape[-2]
		if input_channel!=output_channel or x_width !=inputs_width:
			residual = self.conv2d(residual,output_channel,kernel_size=1,stride=stride[1],relu=False)
		return tf.nn.relu(tf.add(inputs,residual))

	def conv2d(self,inputs,output_channel,kernel_size=3,stride=1,relu=True):
		inputs = tf.layers.conv2d(inputs,filters=output_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,kernel_regularizer=self.regularizer)
		inputs = tf.layers.batch_normalization(inputs,training=self.is_training)
		inputs =  tf.nn.relu(inputs) if relu else inputs
		return inputs

	def forward(self,inputs):
		out = self.conv2d(inputs,16)
		out = self.make_layer(out,[16,32])
		out = self.make_layer(out,[32,64])
		out = self.make_layer(out,[64,64])
		out = tf.layers.average_pooling2d(out,pool_size=8,strides=1)
		out = tf.layers.flatten(out)
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_initializer=self.initializer,kernel_regularizer=self.regularizer)
		softmax_out = tf.nn.softmax(predicts,name='output')
		return predicts,softmax_out

	def make_layer(self,inputs,output_channel):
		stride_2 = output_channel[1] // output_channel[0]
		for i in range(self.stack_num-1):
			inputs = self.residual_block(inputs,output_channel[0])
		inputs = self.residual_block(inputs,output_channel[1],stride=[1,stride_2])
		return inputs


	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		losses+=tf.add_n(l2_reg)
		return losses

'''
layer number :6*stack_num+2

'''

def resnet20(is_training=True,keep_prob=0.5):
	net = Resnet(is_training=is_training,keep_prob=keep_prob,stack_num=3)
	return net 


def resnet32(is_training=True,keep_prob=0.5):
	net = Resnet(is_training=is_training,keep_prob=keep_prob,stack_num=5)
	return net 


def resnet44(is_training=True,keep_prob=0.5):
	net = Resnet(is_training=is_training,keep_prob=keep_prob,stack_num=7)
	return net 


def resnet56(is_training=True,keep_prob=0.5):
	net = Resnet(is_training=is_training,keep_prob=keep_prob,stack_num=9)
	return net 


if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = resnet56()
		data = np.random.randn(64,32,32,3)
		inputs = tf.placeholder(tf.float32,[64,32,32,3])
		predicts,softmax_out = net.forward(inputs)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		init = tf.global_variables_initializer()
		sess = tf.Session(config=config)
		sess.run(init)
		output = sess.run(predicts,feed_dict={inputs:data})
		print(output.shape)
		sess.close()