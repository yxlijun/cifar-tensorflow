import os 

dataset_params = {
	'data_path':'./cifar10_data'
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


def merge_params(FLAGS):
	common_params['batch_size'] = FLAGS.batch_size
	common_params['learning_rate'] = FLAGS.lr
	common_params['moment'] = FLAGS.moment
	common_params['display_step'] = FLAGS.display_step
	common_params['num_epochs'] = FLAGS.num_epochs
	common_params['predict_step'] = FLAGS.predict_step
	return common_params