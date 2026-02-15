# pylint: disable=line-too-long
r"""Default configs for ResNet on Generated ImageNet.

"""

# from scenic.projects.robust_vit.configs.robustness import imagenet_resnet_config
import ml_collections

NUM_CLASSES = 1000

VARIANT = 'R/50'


def get_config(runlocal=''):
  """Returns the base experiment configuration for ImageNet."""

  runlocal = bool(runlocal)
  config = ml_collections.ConfigDict()
  config.experiment_name = 'synthetic_imagenet_resnet'
  # dataset
  config.dataset_name = 'synthetic_imagenet'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  config.dataset_configs.train_size = 1_320_000  # original 1_281_167

  INPUT_RES = 224  # pylint: disable=invalid-name
  RESIZE_RES = int(INPUT_RES * (256 / 224))  # pylint: disable=invalid-name
  LS = 1e-4  # pylint: disable=invalid-name
  augreg = True
  if augreg:
    config.dataset_configs.pp_train = (
        f'decode_jpeg_and_inception_crop({INPUT_RES})|flip_lr|randaug(2, 15)'
        f'|value_range(-1, 1)|onehot({config.dataset_configs.num_classes},'
        f' key="label", key_result="labels", '
        f'on={1.0-LS}, off={LS})|keep("image", "labels")')
  else:
    config.dataset_configs.pp_train = (
        f'decode_jpeg_and_inception_crop({INPUT_RES})|flip_lr|value_range(-1, '
        f'1)|onehot({config.dataset_configs.num_classes},'
        f' key="label", key_result="labels", '
        f'on={1.0-LS}, off={LS})|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      f'decode|resize_small({RESIZE_RES})|'
      f'central_crop({INPUT_RES})|value_range(-1, '
      f'1)|onehot({config.dataset_configs.num_classes},'
      f' key="label", '
      f'key_result="labels")|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2

  # Mixup.
  if augreg:
    config.mixup = ml_collections.ConfigDict()
    config.mixup.bind_to = None
    config.mixup.alpha = 0.4

  config.batch_size = 8192

  # model
  _, layers = VARIANT.split('/')
  config.model_name = 'resnet_classification'
  config.num_filters = 64
  config.num_layers = int(layers)
  config.model_dtype_str = 'float32'

  # training
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.momentum = 0.9
  config.rng_seed = 0
  config.l2_decay_factor = .00005
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 300  # or 90
  config.batch_size = 8192
  config.rng_seed = 0
  config.init_head_bias = -10.0

  # learning rate (used for creating optimizer object).
  steps_per_epoch = config.dataset_configs.train_size // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.1 * config.batch_size / 256
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 7 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr

  # logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 10 * steps_per_epoch
  config.xprof = True  # Profile using xprof
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
