# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf 
import os 
import argparse
from tensorflow.python.framework import graph_util
import numpy as np 


def freeze_graph(model_folder):
	MODEL_DIR = model_folder[:model_folder.rfind('/')]
	MODEL_DIR = os.path.join(MODEL_DIR,'pb')
	if not tf.gfile.Exists(MODEL_DIR):
		tf.gfile.MakeDirs(MODEL_DIR)

	checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
	input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
	output_graph = os.path.join(MODEL_DIR,'frozen_model.pb') #PB模型保存路径
	print(input_checkpoint)
	output_node_names = "output" #原模型输出操作节点的名字
	saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.
	graph = tf.get_default_graph() #获得默认的图
	input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

	data = np.random.randn(1,32,32,3)

	with tf.Session() as sess:
	    saver.restore(sess, input_checkpoint) #恢复图并得到数据
	    print("predictions : ", sess.run("output:0", feed_dict={"input:0": data})) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字
	    output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
	        sess,
	        input_graph_def,
	        output_node_names.split(",") #如果有多个输出节点，以逗号隔开
	    )
	    with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
	        f.write(output_graph_def.SerializeToString()) #序列化输出


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="input ckpt model dir")
    args = parser.parse_args()
    freeze_graph(args.model_folder)


