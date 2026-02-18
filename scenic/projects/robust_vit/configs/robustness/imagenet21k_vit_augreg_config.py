# pylint: disable=line-too-long
r"""Default configs for ViT on ImageNet-21k.


"""
# pylint: enable=line-too-long

import ml_collections

_IMAGENET21K_TRAIN_SIZE = 12743321

NUM_CLASSES = 21843
VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet-21k."""
  runlocal = bool(runlocal)
  config = base_21k_augreg_config(runlocal, VARIANT)
  return config


def base_21k_augreg_config(runlocal, variant):
  """set up the basix config for 21k augreg."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet21k-vit-augreg300'

  # dataset
  config.dataset_name = 'imagenet21k'
  config.data_dtype_str = 'float32'

  config.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr'
      '|randaug(2, 15)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.shuffle_buffer_size = 250_000

  # model
  version, patch = variant.split('/')
  config.model_name = 'vit_multilabel_classification'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}[version]
  config.model.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.1
  config.model.stochastic_depth = 0.1  # Model Regularization
  config.model_dtype_str = 'float32'

  # training
  config.trainer_name = 'fewshot_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.03
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0 if version == 'L' else None
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # learning rate
  steps_per_epoch = _IMAGENET21K_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
  config.lr_configs.total_steps = total_steps
  # config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.5

  # logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 1000
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
