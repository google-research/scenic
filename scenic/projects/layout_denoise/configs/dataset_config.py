"""Implements common configs for layout denoise datasets."""

import ml_collections

TRAIN_SIZE = {
    'rico': 44629,
}

EVAL_SIZE = {
    'rico': 6207,
}


def get_config(data_name='rico', shuffle_buffer_size=10_000, use_inner=True):
  """Returns configs for a datset given its name."""
  dataset_configs = ml_collections.ConfigDict()
  dataset_configs.shuffle_buffer_size = shuffle_buffer_size
  dataset_configs.prefetch_to_device = 5
  dataset_configs.use_inner = use_inner

  if data_name == 'rico':
    # Add path to training files containing tf.Example.
    dataset_configs.train_files = ['/path/to/train_tfexample']
    # Add path to eval (validation) files containing tf.Example.
    dataset_configs.eval_files = ['/path/to/eval_tfexample']

  dataset_configs.dataset_name = data_name
  dataset_configs.task_name = 'layout_denoise'
  dataset_configs.num_train_examples = TRAIN_SIZE[data_name]
  dataset_configs.num_eval_examples = EVAL_SIZE[data_name]
  dataset_configs.data_dtype_str = 'float32'
  return dataset_configs
