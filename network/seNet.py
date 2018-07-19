from __future__ import print_function
from __future__ import division 
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 


class SeNet(object):
	"""docstring for SeNet"""
	def __init__(self, block_num,num_classes=10,is_training=True):
		super(SeNet, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.block_nums = block_num

		self.conv_num = 0

		self.weight_decay = 1e-4
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
		self.initializer = tf.contrib.layers.xavier_initializer()

		self.cardinality = 32
		self.ratio = 16

		self.average_num = 0
		self.fc_num = 0

	def conv2d(self,inputs,out_channel,kernel_size,stride,relu=True):
		out = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		out = tf.layers.batch_normalization(out,training=self.is_training,name='bn_'+str(self.conv_num))
		self.conv_num+=1
		return tf.nn.relu(out) if relu else out


	def se_fc_layer(self,inputs):
		b,h,w,c = inputs.shape
		out = tf.layers.average_pooling2d(inputs,pool_size=(h,w),strides=(h,w),padding='same',name='average_pool_'+str(self.average_num))
		self.average_num+=1
		out = tf.layers.flatten(out)
		out = tf.layers.dense(out,units=(c//self.ratio),activation=tf.nn.relu,kernel_initializer=self.initializer,name='fc_'+str(self.fc_num))
		self.fc_num+=1
		out = tf.layers.dense(out,units=c,activation=tf.sigmoid,kernel_initializer=self.initializer,name='fc_'+str(self.fc_num))
		self.fc_num+=1
		out = tf.reshape(out,(-1,1,1,c))
		out = tf.tile(out,(1,h,w,1))
		return out



	def SE_bottleneck_layer(self,inputs,out_channel,stride):
		'''
			params:
			inputs :net inputs
			out_channel: [list] eg:[128,128,256]
		  	stride: [list] eg:[1,1,2]
		'''
		residual = tf.identity(inputs)
		input_channel = residual.shape[-1]
		input_width = residual.shape[-2]

		out = self.conv2d(inputs,out_channel=out_channel[0],kernel_size=1,stride=stride[0])
		out = self.conv2d(out,out_channel=out_channel[1],kernel_size=3,stride=stride[1])
		out = self.conv2d(out,out_channel=out_channel[2],kernel_size=1,stride=stride[2],relu=False)

		_out = tf.identity(out)
		out = self.se_fc_layer(out)
		out = tf.multiply(_out,out)

		out_width = out.shape[-2]
		if input_channel!=out_channel[2] or input_width !=out_width:
			residual = self.conv2d(residual,out_channel[2],kernel_size=1,stride=stride[2],relu=False)

		return tf.nn.relu(tf.add(out,residual))


	def forward(self,inputs):
		inputs_width = inputs.shape[-1]
		out = self.conv2d(inputs,out_channel=64,kernel_size=7,stride=2)
		out = tf.layers.max_pooling2d(out,pool_size=3,strides=2,padding='same',name='max_pool0')
		out = self.make_layer(out,out_channels=[64,64,256],block_num=self.block_nums[0],downsample=False)
		out = self.make_layer(out,out_channels=[128,128,512],block_num=self.block_nums[1])
		out = self.make_layer(out,out_channels=[256,256,1024],block_num=self.block_nums[2])
		out = self.make_layer(out,out_channels=[512,512,2048],block_num=self.block_nums[3])

		h,w = inputs.shape[1] // 32 ,inputs.shape[2] // 32
		out = tf.layers.average_pooling2d(out,pool_size=(h,w),strides=(h,w),padding='same',name='average_pool_'+str(self.average_num))
		out = tf.layers.flatten(out,name='flatten')
		out = tf.layers.dropout(out,rate=0.5,name='dropout')
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_regularizer=self.regularizer,name='fc_'+str(self.fc_num))
		softmax_out = tf.nn.softmax(predicts,name='output')
		
		return predicts,softmax_out



	def make_layer(self,inputs,out_channels,block_num,downsample=True):
		stride_2 = 2 if downsample else 1
		for i in range(block_num-1):
			inputs = self.SE_bottleneck_layer(inputs,out_channel=out_channels,stride=[1,1,1])
		inputs = self.SE_bottleneck_layer(inputs,out_channel=out_channels,stride=[1,1,stride_2])
		return inputs


	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		losses+=tf.add_n(l2_reg)
		return losses



def SE_Resnet_50():
	net = SeNet([3,4,6,3])
	return net 

def SE_Resnet_101():
	net = SeNet([3,4,23,3])
	return net 


if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = SE_Resnet_50()
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