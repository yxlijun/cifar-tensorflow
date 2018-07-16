from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf 
from config.cfg import dataset_params


_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}



def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record,is_training):
	record_vector = tf.decode_raw(raw_record,tf.uint8)
	label = tf.cast(record_vector[0],tf.int32)
	depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],[_NUM_CHANNELS,_HEIGHT,_WIDTH])

	image = tf.cast(tf.transpose(depth_major,[1,2,0]),tf.float32)
	image = preprocess_image(image,is_training)
	return image,label


def preprocess_image(image, is_training):
	if is_training:
		image = tf.image.resize_image_with_crop_or_pad(image,_HEIGHT+8,_WIDTH+8)
		image = tf.random_crop(image,[_HEIGHT,_WIDTH,_NUM_CHANNELS])
		image = tf.image.random_flip_left_right(image)
	image = tf.image.per_image_standardization(image)
	return image

def input_fn(is_training, data_dir, batch_size, num_epochs=1):
	filenames = get_filenames(is_training,data_dir)
	dataset = tf.data.FixedLengthRecordDataset(filenames,_RECORD_BYTES)
	dataset = dataset.prefetch(buffer_size=batch_size)
	if is_training:
		dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.apply(tf.contrib.data.map_and_batch(
		lambda value: parse_record(value, is_training),
		batch_size=batch_size,
		num_parallel_batches=1,
		drop_remainder=False))	

	dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
	return dataset

if __name__=='__main__':
	dataset = input_fn(True,dataset_params['data_path'],64,135)
	print(dataset)