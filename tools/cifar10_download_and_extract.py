from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from config.cfg import dataset_params

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

parser = argparse.ArgumentParser()


parser.add_argument(
    '--data_dir', type=str, default=dataset_params['data_path'],
    help='Directory to download data and extract the tarball')


def main(_):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(FLAGS.data_dir, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(FLAGS.data_dir)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()