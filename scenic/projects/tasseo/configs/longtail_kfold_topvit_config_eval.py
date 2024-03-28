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
r"""TopViT on long tail dataset with k-fold CV support.

"""
# pylint: disable=line-too-long

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
  """Gets config for training from scratch for all CV fold iterations."""
  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'longtail-topvit-kfold'
  config.trainer_name = 'inference'

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
  config.init_from = ml_collections.ConfigDict()
  config.init_from.xm = (None, None)
  config.batch_size = 8 if runlocal else 256
  config.rng_seed = 42
  config.save_predictions = True

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  pattern_pathnames = ['inv_16', 'inv_3_q21q2', 't_11_19', 't_9_11']
  test_fold_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  # wids corresponding to pattern/test_fold combination.
  wids = [(43892109, i) for i in range(1, 41)]
  test_set_domain = hyper.product([
      hyper.sweep('config.dataset_configs.pattern_pathname', pattern_pathnames),
      hyper.sweep('config.dataset_configs.test_fold_num', test_fold_nums),
  ])
  init_from_domain = hyper.product([
      hyper.sweep('config.init_from.xm', wids),])

  return hyper.zipit([test_set_domain, init_from_domain])
