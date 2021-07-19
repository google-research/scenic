# pylint: disable=line-too-long
r"""Default configs for MNIST classification.

# BEGIN GOOGLE-INTERNAL
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--xm_skip_launch_confirmation --noxm_monitor_on_launch \
--config=third_party/py/scenic/projects/baselines/configs/mnist/mnist_config.py \
--platform=jf_2x2 --cell=tp --xm_resource_alloc=group:brain/dehghani \
--exp_name=mnist
# END GOOGLE-INTERNAL
"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """Returns the base experiment configuration for MNIST."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'mnist'
  # dataset
  config.dataset_name = 'mnist'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'

  # model
  config.model_name = 'fully_connected_classification'
  config.model_dtype_str = 'float32'
  config.hid_sizes = [64, 64]
  # training
  config.trainer_name = 'classification_trainer'

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = 0.1

  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.momentum = 0.9
  config.l2_decay_factor = .0005
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 10
  config.batch_size = 128
  config.rng_seed = 0
  # logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
