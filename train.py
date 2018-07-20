from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config.cfg as cfg 
from solver.solver_cifar import Solver
from data.dataset import cifar_dataset

import argparse 
import tensorflow as tf 


parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=0.1,help='cifar_10 learning_rate')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--moment',type=float,default=0.9,help='sovler moment')
parser.add_argument('--display_step',type=int,default=100,help='show train display')
parser.add_argument('--num_epochs',type=int,default=200,help='train epochs')
parser.add_argument('--predict_step',type=int,default=500,help='predict step')
parser.add_argument('-n','--net',type=str,default='vgg11',choices=cfg.net_style,help='net style')


def main(_):
	print('please choose net from:',cfg.net_style)
	common_params = cfg.merge_params(FLAGS)
	print(common_params)
	net_name = FLAGS.net
	dataset = cifar_dataset(common_params,cfg.dataset_params)
	solver = Solver(net_name,dataset,cfg.common_params,cfg.dataset_params)
	solver.solve()

if __name__=='__main__':
	FLAGS,unknown = parser.parse_known_args()
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)