# pylint: disable=line-too-long
r"""Default configs for ImageNet Robustness evaluation.


# local
# As we load xid from xmanager, this config does not support local run
# See the colab for debugging instruction.
"""
import os  # pylint: disable=unused-import
import ml_collections


def get_base_config(runlocal, xid, wid, image_size=224, ckpt_path=None, evaluate_all_checkpoints=False):
  """Returns the configuration for ImageNet Robustness Evaluation."""
  runlocal = bool(runlocal)
  # parameters
  config = ml_collections.ConfigDict()
  config.xid = xid
  config.wid = wid
  config.image_size = image_size
  config.ckpt_path = ckpt_path  # if not None, used to override the ckpt in xmanager.
  # Example ckpt_path:
  # for legacy jobs
  # ckpt_path = os.path.split(ckpt_path)[0]  # for removes .../1/
  # ckpt_path = path.replace('vit', 'vit_models') # for relocated models

  # user config for robustness evaluation
  config.rng_seed = 0
  config.dataset_name = 'imagenet_variants'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset_name = 'imagenet'

  config.init_from = ml_collections.ConfigDict()
  config.init_from.evaluate_all_checkpoints = evaluate_all_checkpoints
  config.init_from.checkpoint_path = None  # do not change dummy entry

  config.eval_batch_size = 64    # for evaluation use a smaller batch size
  config.batch_size = config.eval_batch_size
  config.trainer_name = 'vit_robustness_evaluator'

  # Needs the following line for using other PP_eval
  # config.dataset_configs.pp_eval = 'default'

  return config


def get_config(config_string):
  """Configuration for trained model."""
  model_zoo = {
      'resnet50':
          get_base_config(False, None, 1),
  }
  return model_zoo[config_string]


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
