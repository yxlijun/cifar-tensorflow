from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import numpy as np 



net_config = {
	'base':128,
	'incre':128,
	'pct33':0.5,
	'freq':2,
	'sr':0.5
}


class SqueezeNet(object):
	"""docstring for SqueezeNet"""
	def __init__(self, net_config,mode='A',num_classes=10):
		super(SqueezeNet, self).__init__()
		self.num_classes = num_classes
		self.conv_num = 0

		self.initializer = tf.contrib.layers.xavier_initializer()

		self.sr = net_config['sr']
		self.base = net_config['base']
		self.incre = net_config['incre']
		self.pct33 = net_config['pct33']
		self.freq = net_config['freq']

		if mode=='A':
			self.make_layer = self.make_layerA
		elif mode =='B':
			self.make_layer = self.make_layerB
		else:
			raise Exception("mode must be A or B")


	def Fiber_module(self,inputs,out_channel):
		sfilter1x1 = self.sr * out_channel
		efilter1x1 = (1-self.pct33) * out_channel
		efilter3x3 = self.pct33 * out_channel
		out = self.conv2d(inputs,sfilter1x1,kernel_size=1,stride=1)
		out_1 = self.conv2d(out,efilter1x1,kernel_size=1,stride=1)
		out_2 = self.conv2d(out,efilter3x3,kernel_size=3,stride=1)
		out = tf.concat([out_1,out_2],axis=-1)
		return out


	def Fiber_moduleB(self,inputs,out_channel):
		resudial = tf.identity(inputs)
		sfilter1x1 = self.sr * out_channel
		efilter1x1 = (1-self.pct33) * out_channel
		efilter3x3 = self.pct33 * out_channel
		out = self.conv2d(inputs,sfilter1x1,kernel_size=1,stride=1)
		out_1 = self.conv2d(out,efilter1x1,kernel_size=1,stride=1,relu=False)
		out_2 = self.conv2d(out,efilter3x3,kernel_size=3,stride=1,relu=False)
		out = tf.concat([out_1,out_2],axis=-1)
		return tf.nn.relu(tf.add(resudial,out))


	def conv2d(self,inputs,out_channel,kernel_size,stride,relu=True):
		out = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,name='conv_'+str(self.conv_num))
		self.conv_num+=1
		return tf.nn.relu(out) if relu else out


	def forward(self,inputs):
		input_width = inputs.shape[-2]

		out = self.conv2d(inputs,out_channel=96,kernel_size=7,stride=2)
		out = tf.layers.max_pooling2d(out,pool_size=3,strides=2,padding='same',name='maxpool_0')
		out = self.make_layer(out)
		out = self.conv2d(out,out_channel=1000,kernel_size=1,stride=1)

		pool_size,stride = (input_width //16),(input_width//16)

		out = tf.layers.average_pooling2d(out,pool_size=(pool_size,pool_size),strides=(stride,stride),name='avg_pool_0')
		out = tf.layers.flatten(out,name='flatten')
		out = tf.layers.dropout(out,rate=0.5,name='dropout')
		predicts = tf.layers.dense(out,units=self.num_classes,name='fc')
		softmax_out = tf.nn.softmax(predicts)

		return predicts,softmax_out

	def make_layerA(self,inputs):
		max_pool_loc = [4,8]
		pool_num = 1
		for i in range(2,10):
			out_channel = self.base+self.incre*((i-2)//self.freq)
			inputs = self.Fiber_module(inputs,out_channel)
			if i in max_pool_loc:
				inputs = tf.layers.max_pooling2d(inputs,pool_size=3,strides=2,padding='same',name='maxpool_'+str(pool_num))
				pool_num+=1
		return inputs


	def make_layerB(self,inputs):
		max_pool_loc = [4,8]
		pool_num = 1
		resudial_loc = [3,5,7,9]
		for i in range(2,10):
			out_channel = self.base+self.incre*((i-2)//self.freq)
			if i in resudial_loc:
				inputs = self.Fiber_moduleB(inputs,out_channel)
			else:
				inputs = self.Fiber_module(inputs,out_channel)
			if i in max_pool_loc:
				inputs = tf.layers.max_pooling2d(inputs,pool_size=3,strides=2,padding='same',name='maxpool_'+str(pool_num))
				pool_num+=1
		return inputs



	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		return losses



def SqueezeNetA():
	net = SqueezeNet(net_config=net_config)
	return net 


def SqueezeNetB():
	net = SqueezeNet(net_config=net_config,mode='B')
	return net

	
if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = SqueezeNetB()
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
