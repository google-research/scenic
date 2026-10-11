# pylint: disable=line-too-long
r"""Default configs for Regularized ViT on ImageNet2012.

"""
# pylint: disable=line-too-long

import ml_collections
from scenic.projects.robust_vit.configs.robustness import common_adaptation
from scenic.projects.robust_vit.configs.robustness import imagenet_vqvit_reg_config

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)
  config = imagenet_vqvit_reg_config.get_base_config(variant=VARIANT, runlocal=runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-regularized_vit'
  # dataset
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()

  config.dataset_configs.dataset = ''  # Set in sweep
  config.dataset_configs.dataset_dir = None  # Set in sweep
  config.dataset_configs.val_split = []  # Set in sweep
  config.dataset_configs.train_split = ''  # Set in sweep
  config.dataset_configs.num_classes = None  # Set in sweep
  config.dataset_configs.pp_train = ''  # Set in sweep
  config.dataset_configs.pp_eval = ''  # Set in sweep
  config.dataset_configs.prefetch_to_device = 2
  # shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 50_000

  # pretrained model info
  config.init_from = ml_collections.ConfigDict()

  # Training.
  config.trainer_name = 'robust_trainer'
  config.optimizer = 'adam_vitonly'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 1e-8
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = None  # we use number of steps
  config.num_training_steps = None  # Set in sweep
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 512

  config.retrain_embed_code = False
  config.use_raw_vqencode = False

  # Learning rate.
  base_lr = 1e-4
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay'
  config.lr_configs.steps_per_cycle = None
  config.lr_configs.warmup_steps = 0  # Set in sweep
  config.lr_configs.total_steps = None
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5000
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval

  config.m = None  # placeholder for randaug strength
  config.l = None  # placeholder for randaug layers

  return config


def get_hyper(hyper):
  """Sweeps over different configs."""
  return hyper.chainit([
      hyper.product([
          common_adaptation.imagenet(
              hyper,
              hres=448,
              lres=384,
              crop='inception_crop',
              steps=20_000,
              warmup=500),
          hyper.sweep('config.lr_configs.base_learning_rate',
                      [0.00001]),
      ]),
  ])
