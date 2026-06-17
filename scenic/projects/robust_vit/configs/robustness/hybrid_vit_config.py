# pylint: disable=line-too-long
r"""Default configs for Hybrid-ViT on JFT.

gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--xm_skip_launch_confirmation --noxm_monitor_on_launch \
--config=third_party/py/scenic/projects/baselines/configs/google/jft/jft_hybrid_vit_config.py
\
--platform=df_8x8 --cell=tp \
--exp_name=R26+Vit-B/32
"""
# pylint: enable=line-too-long

import ml_collections
# from scenic.projects.baselines.configs.google.common import common_fewshot

_JFT_TRAIN_SIZE = 303_021_387

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000
VARIANT = 'R50x1+B/16'

RESNET_NUM_LAYERS = {
    (50, 16): (3, 4, 6, 3),
    (50, 32): (3, 4, 9),
    (26, 32): (2, 2, 2),
}


def get_config():
  """Returns the Hybrid-ViT experiment configuration for JFT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'jft-hybrid-vit-base/16'

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
      '|randaug(2, 15)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')

  # Model.
  resnet_type, version_patch = VARIANT.split('+')
  version, patch = version_patch.split('/')
  resnet_n_layers, width = resnet_type.replace('R', '').split('x')
  patch, resnet_n_layers, width = int(patch), int(resnet_n_layers), int(width)

  config.model_name = 'hybrid_vit_multilabel_classification'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}[version]
  config.model.resnet = ml_collections.ConfigDict()
  config.model.resnet.num_layers = RESNET_NUM_LAYERS[(resnet_n_layers, patch)]
  config.model.resnet.width_factor = width
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (1, 1)
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}[version]
  config.model.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}[version]
  # config.model.representation_size = config.model.hidden_size
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model_dtype_str = 'float32'

  # Training.
  # config.trainer_name = 'fewshot_trainer'
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 4096
  config.rng_seed = 42
  config.init_head_bias = -10.0

  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.5

  # learning rate
  # steps_per_epoch = _JFT_TRAIN_SIZE // config.batch_size
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 4e-4 if version == 'L' else 8e-4
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # # fewshot
  # config.fewshot = common_fewshot.get_config(config.batch_size)
  # config.fewshot.representation_layer = 'pre_logits'
  # config.fewshot.log_eval_steps = 25_000

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
