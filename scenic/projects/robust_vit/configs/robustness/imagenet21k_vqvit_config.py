# pylint: disable=line-too-long
r"""Config file for discrete VQVAE on ImageNet21K dataset.

"""
# pylint: disable=line-too-long

import ml_collections


_IMAGENET21K_TRAIN_SIZE = 12743321
VARIANT = 'B/16'


def get_fewshot_config(batch_size=None, target_resolution=224, resize_resolution=256):
  """Returns a standard-ish fewshot eval configuration."""
  config = ml_collections.ConfigDict()
  config.batch_size = batch_size
  config.representation_layer = 'pre_logits'
  config.log_steps = 25_000
  config.datasets = {
      'birds': ('caltech_birds2011', 'train', 'test'),
      'caltech': ('caltech101', 'train', 'test'),
      'cars': ('cars196:2.1.0', 'train', 'test'),
      'cifar100': ('cifar100', 'train', 'test'),
      'col_hist': ('colorectal_histology', 'train[:2000]', 'train[2000:]'),
      'dtd': ('dtd', 'train', 'test'),
      'imagenet': ('imagenet2012_subset/10pct', 'train', 'validation'),
      'pets': ('oxford_iiit_pet', 'train', 'test'),
      'uc_merced': ('uc_merced', 'train[:1000]', 'train[1000:]'),
  }
  config.pp_train = f'decode|resize({resize_resolution})|central_crop({target_resolution})|value_range(-1,1)'
  config.pp_eval = f'decode|resize({resize_resolution})|central_crop({target_resolution})|value_range(-1,1)'
  config.shots = [1, 5, 10, 25]
  config.l2_regs = [2.0**i for i in range(-10, 20)]
  config.walk_first = ('imagenet', 10)

  return config


def get_base_21k_config(runlocal, variant):
  """set up basic config for 21k training."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet21k-vqvit'

  # dataset
  config.dataset_name = 'imagenet21k'
  config.data_dtype_str = 'float32'

  # model
  version, _ = variant.split('/')
  config.model_name = 'robust_vit_multilabel_classification'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}[version]
  config.model.patches = ml_collections.ConfigDict()
  # config.model.patches.size = [int(patch), int(patch)]
  config.model.patches.grid = [14, 14]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}[version]
  config.model.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.1
  config.model_dtype_str = 'float32'

  # gan
  config.model_class = 'vqgan'
  config.pretrained_image_model = True
  config.perceptual_loss_weight = 0.00005
  config.perceptual_loss_on_logit = True

  config.vqgan_dir = 'path/to/ImageNet21K_VQGANModel'

  config.vqgan = ml_collections.ConfigDict()
  config.vqgan.loss_type = 'non-saturating'
  config.vqgan.architecture = 'basic'
  config.vqgan.g_adversarial_loss_weight = 0.1
  config.vqgan.gradient_penalty = 'r1'  #  r1, none
  config.vqgan.grad_penalty_cost = 10.0

  config.vqvae = ml_collections.ConfigDict()
  config.vqvae.quantizer = 'vq'  #  "gumbel" or "vq"
  config.vqvae.codebook_size = 1024
  config.vqvae.architecture = 'enc_dec_arc'

  config.vqvae.entropy_loss_ratio = 0.1
  config.vqvae.entropy_temperature = 0.01
  config.vqvae.entropy_loss_type = 'softmax'
  config.vqvae.commitment_cost = 0.25

  config.vqvae.filters = 128
  config.vqvae.num_res_blocks = 2
  config.vqvae.channel_multipliers = [1, 1, 2, 2, 4]
  config.vqvae.embedding_dim = 256
  config.vqvae.conv_downsample = False
  config.vqvae.activation_fn = 'swish'
  config.vqvae.norm_type = 'GN'

  config.StyleGANDiscriminator = ml_collections.ConfigDict()
  config.StyleGANDiscriminator.channel_multiplier = 1
  config.StyleGANDiscriminator.blur_resample = False

  config.tau_anneal = ml_collections.ConfigDict()
  config.tau_anneal.tau_max = 1.0
  config.tau_anneal.tau_min = 0.6
  config.tau_anneal.tau_warmup_steps = 0
  config.tau_anneal.tau_decay_steps = 100_000

  # Training.
  config.trainer_name = 'robust_trainer'
  config.optimizer = 'adam_vitonly'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.03
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0 if version == 'L' else None
  config.max_grad_norm = 1.0  # have nan for raw vq.
  config.label_smoothing = None
  config.num_training_epochs = 90
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 42
  config.init_head_bias = -10.0  # -ln(21000)

  config.retrain_embed_code = False
  config.use_raw_vqencode = True

  # Learning rate.
  steps_per_epoch = _IMAGENET21K_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * linear_warmup * linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # fewshot
  config.fewshot = get_fewshot_config(config.batch_size)
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

  config.m = None  # placeholder for randaug strength
  config.l = None  # placeholder for randaug layers
  return config


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)
  config = get_base_21k_config(runlocal, VARIANT)

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
