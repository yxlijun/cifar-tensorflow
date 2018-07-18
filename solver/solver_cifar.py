from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np 
import os 
os.environ['CUDA_VISIBLE_DEVICE'] = '1'

class Solver(object):
	"""docstring for Solver"""
	def __init__(self, net,dataset,common_params):
		super(Solver, self).__init__()
		self.net = net 
		self.dataset = dataset
		self.learning_rate = common_params['learning_rate']
		self.moment = common_params['moment']
		self.batch_size = common_params['batch_size']
		self.height,self.width = common_params['image_size']

		self.display_step = common_params['display_step']
		self.predict_step = common_params['predict_step']

		self.construct_graph()

	def construct_graph(self):
		self.global_step = tf.Variable(0, trainable=False)
		self.images = tf.placeholder(tf.float32, (None, self.height, self.width, 3))
		self.labels = tf.placeholder(tf.int32, None)
		self.predicts,self.softmax_out = self.net.forward(self.images)
		self.total_loss = self.net.loss(self.predicts,self.labels)
		self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.moment).minimize(self.total_loss,global_step=self.global_step)

		correct_pred = tf.equal(tf.argmax(self.softmax_out,1,output_type=tf.int32),self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

	def solve(self):
		train_iterator = self.dataset['train'].make_one_shot_iterator()
		train_dataset = train_iterator.get_next()
		test_iterator = self.dataset['test'].make_one_shot_iterator()
		test_dataset = test_iterator.get_next()

		init = tf.global_variables_initializer()
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		init = tf.global_variables_initializer()
		sess = tf.Session(config=config)
		sess.run(init)
		step = 1
		acc_count = 0
		total_accuracy = 0
		try:
		    while True:
		        images,labels = sess.run(train_dataset)
		        sess.run(self.train_op,feed_dict={self.images:images,self.labels:labels})
		        if step % self.display_step==0:
		        	acc = sess.run(self.accuracy,feed_dict={self.images:images,self.labels:labels})
		        	total_accuracy+=acc 
		        	acc_count+=1
		        	loss = sess.run(self.total_loss,feed_dict={self.images:images,self.labels:labels})
		        	print('Iter step:%d loss:%.4f accuracy:%.4f' %(step,loss,total_accuracy/acc_count))
		        if step % self.predict_step == 0:
		        	test_images,test_labels = sess.run(test_dataset)
		        	print(test_images.shape)
		        	acc = sess.run(self.accuracy,feed_dict={self.images:test_images,self.labels:test_labels})
		        	print('test loss:%.4f' %(acc))
		       	step+=1
		except tf.errors.OutOfRangeError:
		    print("finish training !")
		sess.close()

