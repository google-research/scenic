# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""TopViT fine-tuning from chrm id on long tail dataset with k-fold CV support.

"""
# pylint: disable=line-too-long

import itertools

import ml_collections


_WARMUP_STEPS = 1000
_TOTAL_STEPS = _WARMUP_STEPS + 5000

# NOTE: Currently, VARIANT is used to configure  input, context, and fused
# encoders, so if you want different configs, you should manually change
# them bellow.
VARIANT = 'Ti/4'

FOLD_BREAKDOWN_METADATA = {
    'inv_16': {
        'all_folds': {'num_abnormal': 19, 'num_normal': 3449},
        0: {'num_abnormal': 1, 'num_normal': 362},
        1: {'num_abnormal': 1, 'num_normal': 360},
        2: {'num_abnormal': 1, 'num_normal': 335},
        3: {'num_abnormal': 1, 'num_normal': 327},
        4: {'num_abnormal': 1, 'num_normal': 342},
        5: {'num_abnormal': 4, 'num_normal': 326},
        6: {'num_abnormal': 1, 'num_normal': 340},
        7: {'num_abnormal': 4, 'num_normal': 365},
        8: {'num_abnormal': 4, 'num_normal': 366},
        9: {'num_abnormal': 1, 'num_normal': 326},
    },
    'inv_3_q21q2': {
        'all_folds': {'num_abnormal': 23, 'num_normal': 3469},
        0: {'num_abnormal': 1, 'num_normal': 363},
        1: {'num_abnormal': 3, 'num_normal': 343},
        2: {'num_abnormal': 2, 'num_normal': 320},
        3: {'num_abnormal': 1, 'num_normal': 333},
        4: {'num_abnormal': 4, 'num_normal': 345},
        5: {'num_abnormal': 5, 'num_normal': 315},
        6: {'num_abnormal': 1, 'num_normal': 328},
        7: {'num_abnormal': 2, 'num_normal': 387},
        8: {'num_abnormal': 2, 'num_normal': 352},
        9: {'num_abnormal': 2, 'num_normal': 383},
    },
    't_11_19': {
        'all_folds': {'num_abnormal': 47, 'num_normal': 3497},
        0: {'num_abnormal': 4, 'num_normal': 348},
        1: {'num_abnormal': 11, 'num_normal': 364},
        2: {'num_abnormal': 5, 'num_normal': 318},
        3: {'num_abnormal': 2, 'num_normal': 378},
        4: {'num_abnormal': 5, 'num_normal': 325},
        5: {'num_abnormal': 3, 'num_normal': 325},
        6: {'num_abnormal': 5, 'num_normal': 388},
        7: {'num_abnormal': 3, 'num_normal': 321},
        8: {'num_abnormal': 5, 'num_normal': 341},
        9: {'num_abnormal': 4, 'num_normal': 389},
    },
    't_9_11': {
        'all_folds': {'num_abnormal': 39, 'num_normal': 3513},
        0: {'num_abnormal': 8, 'num_normal': 341},
        1: {'num_abnormal': 3, 'num_normal': 341},
        2: {'num_abnormal': 2, 'num_normal': 322},
        3: {'num_abnormal': 4, 'num_normal': 352},
        4: {'num_abnormal': 1, 'num_normal': 391},
        5: {'num_abnormal': 1, 'num_normal': 352},
        6: {'num_abnormal': 10, 'num_normal': 350},
        7: {'num_abnormal': 1, 'num_normal': 360},
        8: {'num_abnormal': 4, 'num_normal': 358},
        9: {'num_abnormal': 5, 'num_normal': 346},
    },
}


def get_train_num_abnormal(
    pattern_pathname: str,
    test_fold: int,
) -> int:
  return FOLD_BREAKDOWN_METADATA[pattern_pathname]['all_folds'][
      'num_abnormal'] - FOLD_BREAKDOWN_METADATA[pattern_pathname][test_fold][
          'num_abnormal']


def get_train_num_normal(
    pattern_pathname: str,
    test_fold: int,
) -> int:
  return FOLD_BREAKDOWN_METADATA[pattern_pathname]['all_folds'][
      'num_normal'] - FOLD_BREAKDOWN_METADATA[pattern_pathname][test_fold][
          'num_normal']


def get_config(runlocal=''):
  """Gets config for finetuning from chrm_id for all CV fold iterations."""
  _, patch = VARIANT.split('/')
  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'longtail-topvit-finetuning-kfold'
  # Dataset.
  config.dataset_name = 'longtail_baseline'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.chrm_image_shape = (199, 99)
  config.dataset_configs.pattern_pathname = 'inv_16'
  config.dataset_configs.test_fold_num = 0
  config.dataset_configs.num_abnormal = 18
  config.dataset_configs.num_normal = 3087

  # Model.
  config.model_name = 'topological_vit_classification'
  config.model = ml_collections.ConfigDict()
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.0
  config.model_dtype_str = 'float32'
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.hidden_size = 768
  config.model.num_heads = 12
  config.model.mlp_dim = 768
  config.model.num_layers = 16
  # Pretrained model info.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.xm = (36063788, 15)
  config.init_from.checkpoint_path = None

  # Training.
  config.trainer_name = 'transfer_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_steps = _TOTAL_STEPS
  # Log eval summary (heavy due to global metrics.)
  config.log_eval_steps = 10
  # Log training summary (rather light).
  config.log_summary_steps = 10
  config.batch_size = 8 if runlocal else 256
  config.rng_seed = 42
  config.init_head_bias = -10.0
  config.class_balancing = True

  # Learning rate.
  base_lr = 3e-5
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = _TOTAL_STEPS
  config.lr_configs.end_learning_rate = 1e-6
  config.lr_configs.warmup_steps = _WARMUP_STEPS
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  # Evaluation:
  config.global_metrics = [
      'chrom_recall',
      'chrom_precision',
      'chrom_f1',
      'chrom_roc_auc_score',
      'chrom_auc_pr_score',
      'chrom_specificity',
  ]

  if runlocal:
    config.count_flops = False

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  pattern_pathnames = ['inv_16', 'inv_3_q21q2', 't_11_19', 't_9_11']
  test_fold_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  train_fold_num_abnormals = [
      get_train_num_abnormal(pattern_n, fold_n)
      for (pattern_n,
           fold_n) in itertools.product(pattern_pathnames, test_fold_nums)
  ]
  train_fold_num_normals = [
      get_train_num_normal(pattern_n, fold_n)
      for (pattern_n,
           fold_n) in itertools.product(pattern_pathnames, test_fold_nums)
  ]

  domain1 = hyper.product([
      hyper.sweep('config.dataset_configs.pattern_pathname', pattern_pathnames),
      hyper.sweep('config.dataset_configs.test_fold_num', test_fold_nums),
  ])

  domain2 = hyper.sweep('config.dataset_configs.num_abnormal',
                        train_fold_num_abnormals)
  domain3 = hyper.sweep('config.dataset_configs.num_normal',
                        train_fold_num_normals)
  return hyper.zipit([domain1, domain2, domain3])
