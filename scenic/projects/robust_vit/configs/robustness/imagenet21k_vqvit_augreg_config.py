# pylint: disable=line-too-long
r"""Config file for discrete VQVAE on ImageNet21K dataset with RandAug.


"""
# pylint: disable=line-too-long

import ml_collections
from scenic.projects.robust_vit.configs.robustness import imagenet21k_vit_augreg_config
# from scenic.projects.baselines.configs.common import common_fewshot


_IMAGENET21K_TRAIN_SIZE = 12743321
NUM_CLASSES = 21843
VARIANT = 'B/16'


def get_config(runlocal=''):
  config = vq_reg_get_config(runlocal, VARIANT, num_training_epochs=90)
  return config


def vq_reg_get_config(runlocal='', variant='B/16', num_training_epochs=300):
  """Returns the ViT experiment configuration for ImageNet."""
  runlocal = bool(runlocal)
  config = imagenet21k_vit_augreg_config.base_21k_augreg_config(runlocal, variant)

  config.experiment_name = 'imagenet21k-vqvit-augreg'
  config.model_name = 'robust_vit_multilabel_classification'

  # gan
  config.model_class = 'vqgan'
  config.pretrained_image_model = True
  config.perceptual_loss_weight = 0.00005
  config.perceptual_loss_on_logit = True

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


  # 21K codebook
  config.vqvae.codebook_size = 8192
  config.vqgan_dir = 'path/to/ImageNet21K_VQGANModel'

  config.new_vq_version = True

  # Training.
  config.trainer_name = 'robust_trainer'
  config.optimizer = 'adam_vitonly'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.03
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = num_training_epochs
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 42
  config.init_head_bias = -10  # -log(1000)

  config.retrain_embed_code = False
  config.use_raw_vqencode = False

  # Learning rate.
  steps_per_epoch = _IMAGENET21K_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * linear_warmup * cosine_decay'
  config.lr_configs.total_steps = total_steps
  # config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
