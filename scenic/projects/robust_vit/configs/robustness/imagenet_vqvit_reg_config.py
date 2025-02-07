# pylint: disable=line-too-long
r"""Default configs for Discrete Only using VQGAN for Regularized ViT on ImageNet2012.

For Table 1 experiment in https://arxiv.org/abs/2111.10493

"""

# pylint: disable=line-too-long

import ml_collections


_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'B/16'


def get_config(runlocal=''):
  """Configuration for vqvit."""
  runlocal = bool(runlocal)

  config = get_base_config(variant=VARIANT, runlocal=runlocal, num_training_epochs=800, base_lr=0.003)
  config.use_raw_vqencode = False
  config.both_vq_raw = False

  # config.att_add_pos = True # not implemented
  config.model.classifer = 'gap'  # use average pooling helps

  # Exp with cat_pos
  config.cat_pos = False
  config.finetune_embed_code = True
  config.finetune_embed_code_lr_decrease = 100

  config.use_raw_vqencode = False
  config.finetune_embed_code = False

  return config


def get_base_config(variant='B/16', runlocal=False, augreg=True, num_training_epochs=300, base_lr=0.001):
  """Returns the ViT experiment configuration for ImageNet."""

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-regularized_vit'
  # dataset
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  if augreg:
    config.dataset_configs.pp_train = (
        'decode_jpeg_and_inception_crop(224)|flip_lr'
        '|randaug(2, 15)'
        '|value_range(-1, 1)'
        f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
        '|keep("image", "labels")')
  else:
    config.dataset_configs.pp_train = (
        'decode_jpeg_and_inception_crop(224)|flip_lr'
        '|value_range(-1, 1)'
        f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
        '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = variant.split('/')
  config.model_name = 'robust_vit_multilabel_classification'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  # config.model.patches.grid = [14, 14]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}[version]
  config.model.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.1
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'

  # gan
  config.model_class = 'vqgan'
  config.pretrained_image_model = True
  config.perceptual_loss_weight = 0.00005
  config.perceptual_loss_on_logit = True

  config.vqgan_dir = 'path/to/ImageNet_VQGANModel'

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
  # config.optimizer = 'momentum_hp_vitonly' # second option for optimizer
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = num_training_epochs
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096

  config.rng_seed = 42
  config.init_head_bias = -6.9  # -log(1000)

  config.retrain_embed_code = False
  config.use_raw_vqencode = False
  config.latent_gaussian = False
  config.gaussian_std = 0
  config.input_gaussian = False
  config.in_gaussian_std = 0  # for range -1,1, and also random scale.

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  # base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.steps_per_cycle = total_steps
  config.total_steps = total_steps
  config.lr_configs.base_learning_rate = base_lr

  # Mixup.
  if augreg:
    config.mixup = ml_collections.ConfigDict()
    config.mixup.bind_to = None
    config.mixup.alpha = 0.5

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
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
