from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 



class XceptionNet(object):
	"""docstring for XceptionNet"""
	def __init__(self, is_training=True,keep_prob=0.5,num_classes=10):
		super(XceptionNet, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.keep_prob = keep_prob
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
		self.initializer = tf.contrib.layers.xavier_initializer()
		self.conv_num = 0
		self.residual_conv = 0

	def conv2d(self,inputs,output_channel,kernel_size,stride=1):
		inputs = tf.layers.conv2d(inputs,filters=output_channel,kernel_size=kernel_size,strides=stride,padding='same',
					kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn_'+str(self.conv_num))
		self.conv_num+=1
		return tf.nn.relu(inputs)

	def separable_conv2d(self,inputs,output_channel):
		inputs = tf.layers.separable_conv2d(inputs,filters=output_channel,kernel_size=3,padding='same',
					depthwise_initializer=self.initializer,pointwise_initializer=self.initializer,pointwise_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn_'+str(self.conv_num))
		self.conv_num+=1
		return inputs

	def residual_separableConv(self,inputs,num_channels):
		residual = tf.layers.conv2d(inputs,filters=num_channels[-1],kernel_size=1,strides=2,padding='same',
					kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='residual_conv_'+str(self.residual_conv))
		for channel in num_channels:
			inputs = tf.nn.relu(inputs)
			inputs = self.separable_conv2d(inputs,channel)
		inputs = tf.layers.max_pooling2d(inputs,pool_size=3,strides=2,padding='same',name='max_pool_'+str(self.residual_conv))
		self.residual_conv+=1
		return tf.add(inputs,residual)

	def block_separableConv(self,inputs,num_channels,repeat=8):
		for index in range(repeat):
			for channel in num_channels:
				inputs = tf.nn.relu(inputs)
				inputs = self.separable_conv2d(inputs,channel)
		return inputs

	def forward(self,inputs):
		out = self.conv2d(inputs,32,3,2)
		out = self.conv2d(out,64,3)
		out = self.residual_separableConv(out,[128,128])
		out = self.residual_separableConv(out,[256,256])
		out = self.residual_separableConv(out,[728,728])
		out = self.block_separableConv(out,[728,728],repeat=8)
		out = self.residual_separableConv(out,[728,1024])
		out = tf.nn.relu(self.separable_conv2d(out,1536))
		out = tf.nn.relu(self.separable_conv2d(out,2048))
		out = tf.layers.average_pooling2d(out,pool_size=1,strides=1,name='average_pool0')
		out = tf.layers.flatten(out,name='flatten')
		out = tf.layers.dropout(out,rate=self.keep_prob,name='dropout')
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='fc')
		softmax_out = tf.nn.softmax(predicts,name='output')
		return predicts,softmax_out

	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		losses+=tf.add_n(l2_reg)
		return losses



if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = XceptionNet()
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
