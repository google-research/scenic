# pylint: disable=line-too-long
r"""Configs for combining Discrete Representation with Continuous Representation on Regularized ViT on ImageNet2012.

With FusionDim=32, can reproduce Table 1's Ours results in https://arxiv.org/abs/2111.10493

"""
# pylint: disable=line-too-long

from scenic.projects.robust_vit.configs.robustness import imagenet_vqvit_reg_config


_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'S/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)
  config = imagenet_vqvit_reg_config.get_base_config(variant=VARIANT, runlocal=runlocal, num_training_epochs=300, base_lr=0.001)

  config.retrain_embed_code = False
  config.use_raw_vqencode = False
  config.is_fusion = True

  config.finetune_embed_code = True
  config.finetune_embed_code_lr_decrease = 100

  config.fusion_dim = 32

  # Learning rate.
  base_lr = 0.001
  config.lr_configs.base_learning_rate = base_lr
  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
