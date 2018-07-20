from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import os 
import argparse
import numpy as np 

from data.dataset import cifar_dataset_test
import config.cfg as cfg 
from network.vgg import vgg11,vgg13,vgg16,vgg19


parser = argparse.ArgumentParser()
parser.add_argument('-n','--net',type=str,default='vgg11',choices=cfg.net_style,help='net style')



def main_ckpt(_):
	model_folder = os.path.join(cfg.dataset_params['model_path'],FLAGS.net,'ckpt')
	checkpoint = tf.train.get_checkpoint_state(model_folder)
	input_checkpoint = checkpoint.model_checkpoint_path
	saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) 

	dataset = cifar_dataset_test(cfg.dataset_params)
	test_iterator = dataset.make_one_shot_iterator()
	test_Loader = test_iterator.get_next()
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	saver.restore(sess, input_checkpoint)
	images,labels = sess.run(test_Loader)
	total_count = 0
	count = 1
	for i in range(0,len(images),50):
		image = images[i:i+50,:,:,:]
		predict = sess.run(cfg.graph_node['output'],feed_dict={cfg.graph_node['input']:image,cfg.graph_node['is_training']:False,cfg.graph_node['keep_prob']:1.0})
		correct_pred = tf.equal(tf.argmax(predict,1,output_type=tf.int32),labels[i:i+50])
		acc = tf.reduce_sum(tf.cast(correct_pred,tf.float32))
		total_count+=sess.run(acc)
		accuracy = total_count/(count*50)
		print("test samples:%d,accuracy:%d/%d = %.4f " %(count*50,total_count,count*50,accuracy))
		count+=1
	sess.close()

def main_pb(_):
	model_folder = os.path.join(cfg.dataset_params['model_path'],FLAGS.net,'pb')
	model_name = os.path.join(model_folder,'frozen_model.pb')
	if not tf.gfile.Exists(model_name):
		raise ValueError('model file not exists')
	with tf.gfile.GFile(model_name,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		with tf.Graph().as_default() as graph:
			tf.import_graph_def(graph_def,name='test')
			inputs = graph.get_tensor_by_name('test/'+cfg.graph_node['input'])
			_predict = graph.get_tensor_by_name('test/'+cfg.graph_node['output'])

	dataset = cifar_dataset_test(cfg.dataset_params)
	test_iterator = dataset.make_one_shot_iterator()
	test_Loader = test_iterator.get_next()

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	sess1 = tf.Session(config=config)
	images,labels = sess1.run(test_Loader)

	sess = tf.Session(config=config,graph=graph)
	total_count = 0
	for i in range(0,len(images),50):
		image = images[i:i+50,:,:,:]
		predict = sess.run(_predict,feed_dict={inputs:image})
		correct_pred = tf.equal(tf.argmax(predict,1,output_type=tf.int32),labels[i:i+50])
		acc = tf.reduce_sum(tf.cast(correct_pred,tf.float32))
		total_count+=sess1.run(acc)
		accuracy = total_count/(i+1)
		print("test samples:%d,accuracy:%d/%d = %.4f " %(i+1,total_count,i+1,accuracy))

	sess.close()
	sess1.close()


if __name__=='__main__':
	FLAGS,unknown = parser.parse_known_args()
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main_ckpt)