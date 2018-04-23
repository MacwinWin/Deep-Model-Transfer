r"""Converts your own dataset to TFRecords of TF-Example protos.

This module reads the files 
that make up the data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils


# The number of images in the validation set.
#_NUM_VALIDATION = 180
PERCENT_VALIDATION = 2.5

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = None


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, dataset_name):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  dataset_root = os.path.join(dataset_dir, dataset_name)
  print('processing data in [%s] :' % dataset_root)
  directories = []
  class_names = []
  for filename in os.listdir(dataset_root):
    path = os.path.join(dataset_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, dataset_name, split_name, shard_id):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      dataset_name, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name,
                     filenames,
                     class_names_to_ids,
                     dataset_dir,
                     dataset_name):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, dataset_name, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()



def _dataset_exists(dataset_dir, dataset_name, split_name):
  for shard_id in range(_NUM_SHARDS):
    output_filename = _get_dataset_filename(
        dataset_dir, dataset_name, split_name, shard_id)
    if not tf.gfile.Exists(output_filename):
      return False
  return True


def run(dataset_dir, dataset_name='dataset'):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)


  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir,
                                     dataset_name)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  #number_validation = len(photo_filenames) * PERCENT_VALIDATION //100
  number_validation = 1000
  print(' total pics number %d' % len(photo_filenames))
  print(' valid number: %d' % number_validation)
  training_filenames = photo_filenames[number_validation:]
  validation_filenames = photo_filenames[:number_validation]

  # First, convert the training and validation sets.
  global _NUM_SHARDS
  _NUM_SHARDS  = len(training_filenames) // 1024
  _NUM_SHARDS = _NUM_SHARDS if _NUM_SHARDS else 1
  if _dataset_exists(dataset_dir, dataset_name, 'train'):
    print('Dataset files already exist. Exiting without re-creating them.')
    return
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir, dataset_name=dataset_name)
  _NUM_SHARDS = len(validation_filenames) // 1024
  _NUM_SHARDS = _NUM_SHARDS if _NUM_SHARDS else 1
  if _dataset_exists(dataset_dir, dataset_name, 'validation'):
    print('Dataset files already exist. Exiting without re-creating them.')
    return
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir, dataset_name=dataset_name)

  # write dataset info
  dataset_info = "label:%d\ntrain:%d\nvalidation:%d" % (
                      len(class_names),
                      len(training_filenames),
                      len(validation_filenames))
  dataset_info_file_path = os.path.join(dataset_dir, dataset_name + '.info')
  with open(dataset_info_file_path, 'w') as f:
    f.write(dataset_info)
    f.flush()


  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

#  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the dataset!')

