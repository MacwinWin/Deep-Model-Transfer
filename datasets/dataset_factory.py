"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import custom


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  return custom.get_split(
      split_name,
      dataset_dir,
      name,
      file_pattern,
      reader)

