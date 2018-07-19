import os 

dataset_params = {
	'data_path':'./cifar10_data',
	'model_path':'./model/train'
}

common_params = {
	'batch_size':64,
	'image_size':(32,32),
	'learning_rate':0.01,
	'moment':0.9,
	'display_step':10,
	'num_epochs':200,
	'predict_step':500
}


graph_node = {
	'input':'input:0',
	'output':'output:0'
}

net_style = ['vgg11','vgg13','vgg16','vgg19',
			'resnet20','resnet32','resnet44','resnet56',
			'XceptionNet',
			'MobileNet',
			'DensetNet40_12','DenseNet100_12','DenseNet100_24','DenseNetBC100_12','DenseNetBC250_24','DenseNetBC190_40',
			'ResNext50','ResNext101',
			'SqueezeNetA','SqueezeNetB',
			'SE_Resnet_50','SE_Resnet_101']


def merge_params(FLAGS):
	common_params['batch_size'] = FLAGS.batch_size
	common_params['learning_rate'] = FLAGS.lr
	common_params['moment'] = FLAGS.moment
	common_params['display_step'] = FLAGS.display_step
	common_params['num_epochs'] = FLAGS.num_epochs
	common_params['predict_step'] = FLAGS.predict_step
	return common_params