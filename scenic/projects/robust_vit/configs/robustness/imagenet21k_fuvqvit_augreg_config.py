# pylint: disable=line-too-long
r"""Configs for pretraining full model (discrete + continuous) Regularized ViT on ImageNet21K.

"""
# pylint: disable=line-too-long

from scenic.projects.robust_vit.configs.robustness import imagenet21k_vqvit_augreg_config


# _IMAGENET21K_TRAIN_SIZE = 12_640_921
_IMAGENET21K_TRAIN_SIZE = 12743321
NUM_CLASSES = 21843
VARIANT = 'B/16'


def get_config(runlocal=''):
  """Defines the config for fusion model."""
  config = imagenet21k_vqvit_augreg_config.vq_reg_get_config(runlocal, VARIANT, num_training_epochs=300)

  # 21K codebook
  config.vqvae.codebook_size = 8192

config.vqgan_dir = 'path/to/ImageNet21K_VQGANModel'

  config.new_vq_version = True

  config.retrain_embed_code = False
  config.use_raw_vqencode = False
  config.is_fusion = True

  config.finetune_embed_code = True
  config.finetune_embed_code_lr_decrease = 100

  config.fusion_dim = 32

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
