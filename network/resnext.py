from __future__ import print_function
from __future__ import division 
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 


class resNext(object):
	"""docstring for resNext"""
	def __init__(self, block_num,num_classes=10,is_training=True):
		super(resNext, self).__init__()
		self.num_classes = num_classes
		self.is_training = is_training
		self.block_nums = block_num

		self.conv_num = 0

		self.weight_decay = 1e-4
		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
		self.initializer = tf.contrib.layers.xavier_initializer()

		self.cardinality = 32

	def bottleneck_layer(self,inputs,out_channel,stride):
		residual = tf.identity(inputs)
		input_channel = residual.shape[-1]
		input_width = residual.shape[-2]

		out = self.conv2d(inputs,out_channel=out_channel[0],kernel_size=1,stride=stride[0])
		out = self.group_conv2d(out,out_channel=out_channel[1],stride=stride[1])
		out = self.conv2d(out,out_channel=out_channel[2],kernel_size=1,stride=stride[2],relu=False)

		out_width = out.shape[-2]
		if input_channel!=out_channel[2] or input_width !=out_width:
			residual = self.conv2d(residual,out_channel[2],kernel_size=1,stride=stride[2],relu=False)

		return tf.nn.relu(tf.add(out,residual))



	def variable_with_weight_deacy(self,name,shape,wd):
		var = tf.get_variable(name,shape,initializer=self.initializer,dtype=tf.float32)
  		if wd is not None:
  			weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
  			tf.add_to_collection('losses',weight_decay)
  		return var 



	def group_conv2d(self,inputs,out_channel,stride):
		scope = 'conv_'+str(self.conv_num)
		in_channel = inputs.shape[-1] // self.cardinality
		out_subchannel = out_channel // self.cardinality
		with tf.variable_scope(scope) as scope:
			filters = self.variable_with_weight_deacy('weights',[3,3,in_channel,out_channel],wd=self.weight_decay)
			group_out = []
			for index in range(self.cardinality):
				out = tf.nn.conv2d(inputs[:,:,:,index*in_channel:(index+1)*in_channel],filter=filters[:,:,:,index*out_subchannel:(index+1)*out_subchannel],strides=[1,stride,stride,1],padding='SAME')
				group_out.append(out)
			group_out = tf.concat(group_out,axis=-1,name='conv_'+str(self.conv_num))
			out = tf.layers.batch_normalization(group_out,training=self.is_training,name='bn_'+str(self.conv_num))
			self.conv_num+=1
			return tf.nn.relu(out)


	def conv2d(self,inputs,out_channel,kernel_size,stride,relu=True):
		out = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,kernel_regularizer=self.regularizer,name='conv_'+str(self.conv_num))
		out = tf.layers.batch_normalization(out,training=self.is_training,name='bn_'+str(self.conv_num))
		self.conv_num+=1
		return tf.nn.relu(out) if relu else out


	def forward(self,inputs):
		input_width = inputs.shape[-2]
		out = self.conv2d(inputs,out_channel=64,kernel_size=7,stride=2)
		out = tf.layers.max_pooling2d(out,pool_size=3,strides=2,padding='same',name='maxpool_0')
		out = self.make_layer(out,[128,128,256],self.block_nums[0],downsample=False)
		out = self.make_layer(out,[256,256,512],self.block_nums[1])
		out = self.make_layer(out,[512,512,1024],self.block_nums[2])
		out = self.make_layer(out,[1024,1024,2048],self.block_nums[3])

		pool_size,stride = (input_width //32),(input_width//32)

		out = tf.layers.average_pooling2d(out,pool_size=(pool_size,pool_size),strides=(stride,stride),name='avg_pool')
		out = tf.layers.flatten(out,name='flatten')
		out = tf.layers.dropout(out,rate=0.5,name='dropout')
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_regularizer=self.regularizer,name='fc')
		softmax_out = tf.nn.softmax(predicts,name='output')
		
		return predicts,softmax_out


	def make_layer(self,inputs,output_channel,block_num,downsample=True):
		stride_2 = output_channel[2] // output_channel[0] if downsample else 1
		for num in range(block_num-1):
			inputs = self.bottleneck_layer(inputs,out_channel=output_channel,stride=[1,1,1])
		inputs = self.bottleneck_layer(inputs,out_channel=output_channel,stride=[1,1,stride_2])
		return inputs



	def loss(self,predicts,labels):
		losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels,predicts))
		l2_reg = tf.losses.get_regularization_losses()
		lr_reg2 = tf.get_collection('losses')
		losses+=tf.add_n(l2_reg)
		losses+=tf.add_n(lr_reg2)
		return losses



def ResNext50():
	'''
		stage1: 1x1x128->group conv->3x3x128 cardinality 32 ->1x1x256
		stage2: 1x1x256->group conv->3x3x256 cardinality 32 ->1x1x512
		stage3: 1x1x512->group conv->3x3x512 cardinality 32 ->1x1x1024
		stage4: 1x1x1024->group conv->3x3x1024 cardinality 32 ->1x1x2048
	'''
	net = resNext([3,4,6,3])
	return net 



def ResNext101():
	net = resNext([3,4,23,3])
	return net 



if __name__=='__main__':
	with tf.device('/cpu:0'):
		net = ResNext50()
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
