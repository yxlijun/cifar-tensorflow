from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import numpy as np 


class DenseNet(object):
	"""docstring for DenseNet"""
	def __init__(self, k = 12,L = 40,base = True,num_classes=10,is_training=True,):
		super(DenseNet, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.k = k
		self.base = base
		self.conv_num = 0
		self.pool_num = 0
		self.weight_decay = 1e-4
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
		self.initializer = tf.contrib.layers.xavier_initializer()

		self.per_block_num = (L - 4)//3 if self.base else (L - 4)//6

	def bottle_block(self,inputs):
		internel_out = tf.identity(inputs)
		for i in range(self.per_block_num):
			inputs = self.bottle_conv2d(internel_out,out_channel=self.k,kernel_size=3)
			internel_out = tf.concat((inputs,internel_out),axis=-1)
		return internel_out



	def transition_layer(self,inputs):
		out_channel = inputs.shape[-1] if self.base else inputs.shape[-1]//2
		out = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn'+str(self.conv_num))
		out = tf.nn.relu(out)
		out = self.conv2d(out,out_channel=out_channel,kernel_size=1)
		out = tf.layers.max_pooling2d(out,pool_size=2,strides=2,padding='same',name='pool_'+str(self.pool_num))
		self.pool_num+=1
		return out 

	def bottle_conv2d(self,inputs,out_channel,kernel_size=3):
		out = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn'+str(self.conv_num))
		out = tf.nn.relu(out)
		if self.base:
			out = self.conv2d(out,out_channel=out_channel,kernel_size = kernel_size)
		else:
			out = self.conv2d(out,out_channel=4*out_channel,kernel_size = 1)
			out = tf.layers.batch_normalization(out,training=self.is_training,name='bn'+str(self.conv_num))
			out = tf.nn.relu(out)
			out = self.conv2d(out,out_channel=out_channel,kernel_size = kernel_size)
		return out

	def conv2d(self,inputs,out_channel,kernel_size=3):
		out = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,padding='same',
				kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		self.conv_num+=1
		return out 

	def forward(self,inputs):
		out = self.conv2d(inputs,out_channel=2*self.k,kernel_size=3)
		out = self.bottle_block(out)
		out = self.transition_layer(out)
		out = self.bottle_block(out)
		out = self.transition_layer(out)
		out = self.bottle_block(out)
		out = tf.layers.average_pooling2d(out,pool_size=8,strides=8,padding='same')
		out = tf.layers.flatten(out)
		out = tf.layers.dropout(out,rate=0.5)
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_regularizer=self.regularizer,name='fc')
		softmax_out = tf.nn.softmax(predicts,name='output')
		return predicts,softmax_out

	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		losses+=tf.add_n(l2_reg)
		return losses


def DensetNet40_12():
	net = DenseNet(k=12,L = 40)
	return net 


def DenseNet100_12():
	net = DenseNet(k=12,L=100)
	return net  

def DenseNet100_24():
	net = DenseNet(k=24,L = 100)
	return net 

def DenseNetBC100_12():
	net = DenseNet(k = 12,L = 100,base=False)
	return net 

def DenseNetBC250_24():
	net = DenseNet(k = 24,L = 250,base = False)
	return net 

def DenseNetBC190_40():
	net = DenseNet(k = 40,L = 190,base = False)
	return net 





if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = DenseNetBC190_40()
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

