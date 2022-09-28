r"""Configs for training a MTV(B/2) with CVA on Kinetics-400.

"""

import ml_collections
from scenic.projects.mfp.configs import common_datasets as datasets
from scenic.projects.mtv import config_utils


# Replace with the actual dataset size.
KINETICS_400_TRAIN_SIZE = 0
KINETICS_400_VAL_SIZE = 0
KINETICS_400_TEST_SIZE = 0
MODEL_VARIANT = 'mvit_T/8+B/2'


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()

  # Dataset.
  config, num_train_samples = datasets.use_kinetics400(config, 16, 2)
  config.experiment_name = f'k400_mtv_mvit_{MODEL_VARIANT}'

  # Model.
  config.model_name = 'mtv_multiclass_classification'
  config.model = ml_collections.ConfigDict()
  config.model.classifier = 'token'
  config.model.dropout_rate = 0.0
  config.model.attention_dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model.view_configs = config_utils.parse_view_configs(MODEL_VARIANT)
  config.model.cross_view_fusion = ml_collections.ConfigDict({
      'type': 'cross_view_attention',
      'fuse_in_descending_order': True,
      'use_query_config': True,
      'fusion_layers': (3, 8),
  })
  config.model.global_encoder_config = ml_collections.ConfigDict({
      'num_heads': 8,
      'mlp_dim': 3072,
      'num_layers': 12,
      'hidden_size': 768,
      'merge_axis': 'channel',
  })
  config.model.temporal_encoding_config = None

  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_type = 'mvit'
  config.init_from.xm = [
      (43214568, 1),  # MViT-T JFT300m stride 4
      (41348301, 1),  # MViT-B JFT300m stride 4
      # (41354730, 1),  # MViT-L JFT300m stride 4
      # (41661802, 1),  # MViT-H JFT300m stride 4
  ]

  # Training.
  config.trainer_name = 'mtv_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.explicit_weight_decay = 5e-2  # This corresponds to Adam-W
  config.max_grad_norm = 1.0
  config.num_training_epochs = 200
  config.batch_size = 256
  config.rng_seed = 0

  # Learning rate.
  batch_size_ref = config.get_ref('batch_size')
  num_epochs_ref = config.get_ref('num_training_epochs')
  steps_per_epoch = num_train_samples // batch_size_ref
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = num_epochs_ref * steps_per_epoch
  config.lr_configs.steps_per_cycle = num_epochs_ref * steps_per_epoch
  config.lr_configs.end_learning_rate = 1e-6
  config.lr_configs.warmup_steps = 5 * steps_per_epoch  # More works better!
  config.lr_configs.base_learning_rate = 1.e-5 * batch_size_ref / 256

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  return config


