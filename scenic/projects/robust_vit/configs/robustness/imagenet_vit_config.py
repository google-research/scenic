# pylint: disable=line-too-long
r"""Default configs for vanilla ViT on ImageNet2012.

"""
# pylint: disable=line-too-long

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000
VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""
  runlocal = bool(runlocal)
  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-vit'
  # dataset
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000

  INPUT_RES = 224  # pylint: disable=invalid-name
  RESIZE_RES = int(INPUT_RES * (256 / 224))  # pylint: disable=invalid-name
  LS = 1e-4  # pylint: disable=invalid-name
  config.dataset_configs.pp_train = (
      f'decode_jpeg_and_inception_crop({INPUT_RES})|flip_lr|value_range(-1, '
      f'1)|onehot({config.dataset_configs.num_classes},'
      f' key="label", key_result="labels", '
      f'on={1.0-LS}, off={LS})|keep("image", '
      f'"labels")')  # pylint: disable=line-too-long
  config.dataset_configs.pp_eval = (
      f'decode|resize_small({RESIZE_RES})|'
      f'central_crop({INPUT_RES})|value_range(-1, '
      f'1)|onehot({config.dataset_configs.num_classes},'
      f' key="label", '
      f'key_result="labels")|keep("image", '
      f'"labels")')  # pylint: disable=line-too-long
  config.dataset_configs.prefetch_to_device = 2

  # model
  version, patch = VARIANT.split('/')
  config.model_name = 'vit_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  # config.model.patches.grid = [14, 14]  ####
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}[version]
  config.model.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.1
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.use_rng_tmp = False

  # training
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.3
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 300  # was 90
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # learning rate (used for creating optimizer object).
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 3e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5000
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  config.count_flops = False

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
