from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import numpy as np 

class MobileNet(object):
	"""docstring for MobileNet"""
	def __init__(self, num_classes=10,is_training=True):
		super(MobileNet, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.conv_num = 0
		self.weight_decay = 5e-4
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
		self.initializer = tf.contrib.layers.xavier_initializer()


	def conv2d(self,inputs,out_channel,kernel_size=1,stride=1):
		inputs = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn'+str(self.conv_num))
		self.conv_num+=1
		return tf.nn.relu(inputs)


  	def _variable_with_weight_decay(self, name, shape,wd):
  		var = tf.get_variable(name,shape,initializer=self.initializer,dtype=tf.float32)
  		if wd is not None:
  			weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
  			tf.add_to_collection('losses',weight_decay)
  		return var 

	def depthwise_conv2d(self,inputs,kernel_size,stride=1):
		scope = 'conv_'+str(self.conv_num)
		with tf.variable_scope(scope) as scope:
			kernel = self._variable_with_weight_decay('weight',shape=kernel_size,wd = self.weight_decay)
			depthwise = tf.nn.depthwise_conv2d(inputs,kernel,[1,stride,stride,1],padding='SAME')
			biases = tf.get_variable('biases',depthwise.shape[3],initializer=tf.zeros_initializer)
			inputs = tf.nn.bias_add(depthwise, biases)
			inputs = tf.layers.batch_normalization(inputs,training=self.is_training,name='bn'+str(self.conv_num))
			inputs = tf.nn.relu(inputs)
		self.conv_num+=1
		return inputs

	def separable_conv2d(self,inputs,out_channel,kernel_size,stride=1):
		inputs = self.depthwise_conv2d(inputs,kernel_size=kernel_size,stride=stride)
		inputs = self.conv2d(inputs,out_channel)
		return inputs

	def forward(self,inputs):
		out = self.conv2d(inputs,out_channel=32,kernel_size=3,stride=2)
		out = self.separable_conv2d(out,out_channel=64,kernel_size=[3,3,32,1],stride=1)
		out = self.separable_conv2d(out,out_channel=128,kernel_size=[3,3,64,1],stride=2)
		out = self.separable_conv2d(out,out_channel=128,kernel_size=[3,3,128,1],stride=1)
		out = self.separable_conv2d(out,out_channel=256,kernel_size=[3,3,128,1],stride=2)
		out = self.separable_conv2d(out,out_channel=256,kernel_size=[3,3,256,1],stride=1)
		out = self.separable_conv2d(out,out_channel=512,kernel_size=[3,3,256,1],stride=2)
		out = self.make_layer(out,repeat=5)
		out = self.separable_conv2d(out,out_channel=1024,kernel_size=[3,3,512,1],stride=2)
		out = self.separable_conv2d(out,out_channel=1024,kernel_size=[3,3,1024,1],stride=1)
		out = tf.layers.average_pooling2d(out,pool_size=1,strides=1)
		out = tf.layers.flatten(out)
		out = tf.layers.dropout(out,rate=0.5)
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_regularizer=self.regularizer,name='fc')
		softmax_out = tf.nn.softmax(predicts,name='output')
		return predicts,softmax_out

	def make_layer(self,inputs,repeat=5):
		for i in range(repeat):
			inputs = self.separable_conv2d(inputs,out_channel=512,kernel_size=[3,3,512,1],stride=1)
		return inputs

	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		lr_reg2 = tf.get_collection('losses')
		losses+=tf.add_n(l2_reg)
		losses+=tf.add_n(lr_reg2)
		return losses



if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = MobileNet()
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

