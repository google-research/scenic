# pylint: disable=line-too-long
r"""Default configs for Mixer on Imagenet.

"""
# pylint: enable=line-too-long

import ml_collections
from scenic.projects.baselines.configs.google.common import common_fewshot

_JFT_TRAIN_SIZE = 303_021_387
_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000
VARIANT = 'B/16'


def get_config():
  """Returns the Mixer experiment configuration for JFT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'jft-mixer-base/16'

  # Dataset.
  # config.dataset_name = 'jft'
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000

  config.dataset_configs.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr'
      '|randaug(2, 10)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')

  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.5

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'mixer_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}[version]
  config.model.patch_size = [int(patch), int(patch)]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}[version]
  config.model.channels_mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096
  }[version]
  config.model.sequence_mlp_dim = {
      'Ti': 96,
      'S': 192,
      'B': 384,
      'L': 512
  }[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}[version]
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  # Training.
  config.trainer_name = 'fewshot_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.03
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 4096
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # learning rate
  steps_per_epoch = _JFT_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001 if version == 'L' else 0.003
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # fewshot
  config.fewshot = common_fewshot.get_config(config.batch_size)
  config.fewshot.representation_layer = 'pre_logits'
  config.fewshot.log_eval_steps = 25_000

  # Logging.
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5000
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
