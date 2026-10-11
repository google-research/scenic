# pylint: disable=line-too-long
r"""Configs for finetuning the given pretrained full (discrete + continuous) model on ImageNet1K.

The ImageNet21K pretrained model is a Regularized ViT that uses both VQGAN based discrete representation and continuous representation (using imagenet21k_fuvqvit_augreg_config.py). VQGAN is trained on ImageNet21K.


"""
# pylint: disable=line-too-long

import ml_collections
from scenic.projects.robust_vit.configs.robustness import common_adaptation

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

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

  # Model.
  version, patch = VARIANT.split('/')
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

  # pretrained model info
  config.init_from = ml_collections.ConfigDict()

  # 21K codebook
  config.vqvae.codebook_size = 8192
  config.vqgan_dir = 'path/to/ImageNet21K_VQGANModel'

  config.new_vq_version = True

  config.retrain_embed_code = False
  config.use_raw_vqencode = False
  config.is_fusion = True

  config.finetune_embed_code = True
  config.finetune_embed_code_lr_decrease = 100

  config.init_from.checkpoint_path = None

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
  config.batch_size = 8 if runlocal else 512  # 256  #
  config.rng_seed = 42
  config.init_head_bias = -6.9  # -log(1000)

  config.retrain_embed_code = False
  config.use_raw_vqencode = False

  # Learning rate.
  # steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  base_lr = 1e-5
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
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
              # hres=522,
              # lres=448,
              # hres=597,
              # lres=512,
              crop='inception_crop',
              steps=20_000,
              warmup=500,
              randaug=False),
          hyper.sweep('config.lr_configs.base_learning_rate',
                      [1e-4]),
          # 0.003, 0.001, 0.0003,
          #        0.0001,0.00001,0.00003
      ]),
  ])
