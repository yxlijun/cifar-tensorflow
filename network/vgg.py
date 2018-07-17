from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VggNet(object):
	"""docstring for VggNet"""
	def __init__(self, vggname,num_classes=10):
		super(VggNet, self).__init__()
		self.vggname = vggname 
		self.num_classes = num_classes

	def forward(self,input):
		regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
		out = self.make_layer(input,cfg[self.vggname],regularizer)
		out = tf.layers.flatten(out,name='flatten')
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_regularizer=regularizer,name='fc_1')
		softmax_out = tf.nn.softmax(predicts)
		return predicts,softmax_out


	def make_layer(self,inputs,netparam,regularizer):
		pool_num = 0
		conv_num = 0
		for param in netparam:
			if param=='M':
				inputs = tf.layers.max_pooling2d(inputs,pool_size=2,strides=2,padding='same',name='pool_'+str(pool_num))
				pool_num+=1
			else:
				inputs = tf.layers.conv2d(inputs,filters=param,kernel_size=3,padding='same',activation=tf.nn.relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=regularizer,name='conv_'+str(conv_num))
				conv_num+=1
		inputs = tf.layers.average_pooling2d(inputs,pool_size=1,strides=1)
		return inputs

	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		losses+=tf.add_n(l2_reg)
		return losses


def vgg11():
	net = VggNet(vggname='VGG11')
	return net 


def vgg13():
	net = VggNet(vggname='VGG13')
	return net 


def vgg16():
	net = VggNet(vggname='VGG16')
	return net 


def vgg19():
	net = VggNet(vggname='VGG19')
	return net 


if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = vgg16()
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
