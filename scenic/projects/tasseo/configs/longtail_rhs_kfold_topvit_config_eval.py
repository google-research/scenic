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
    't_11_19': {
        'all_folds': {'num_abnormal': 48, 'num_normal': 3508},
        0: {'num_abnormal': 1, 'num_normal': 341},
        1: {'num_abnormal': 6, 'num_normal': 361},
        2: {'num_abnormal': 3, 'num_normal': 325},
        3: {'num_abnormal': 4, 'num_normal': 382},
        4: {'num_abnormal': 3, 'num_normal': 323},
        5: {'num_abnormal': 11, 'num_normal': 339},
        6: {'num_abnormal': 5, 'num_normal': 388},
        7: {'num_abnormal': 4, 'num_normal': 326},
        8: {'num_abnormal': 6, 'num_normal': 342},
        9: {'num_abnormal': 5, 'num_normal': 381},
    },
    't_9_11': {
        'all_folds': {'num_abnormal': 68, 'num_normal': 3559},
        0: {'num_abnormal': 3, 'num_normal': 339},
        1: {'num_abnormal': 2, 'num_normal': 341},
        2: {'num_abnormal': 4, 'num_normal': 325},
        3: {'num_abnormal': 2, 'num_normal': 362},
        4: {'num_abnormal': 12, 'num_normal': 398},
        5: {'num_abnormal': 13, 'num_normal': 349},
        6: {'num_abnormal': 9, 'num_normal': 369},
        7: {'num_abnormal': 1, 'num_normal': 346},
        8: {'num_abnormal': 17, 'num_normal': 390},
        9: {'num_abnormal': 5, 'num_normal': 340},
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
  config.experiment_name = 'longtail_rhs-topvit-kfold'
  config.trainer_name = 'inference'

  # Dataset.
  config.dataset_name = 'longtail_rhs_baseline'
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
  config.save_predictions_on_cns = True

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  pattern_pathnames = ['t_11_19', 't_9_11']
  test_fold_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  # wids corresponding to pattern/test_fold combination.
  wids = [(43932067, i) for i in range(1, 41)]
  test_set_domain = hyper.product([
      hyper.sweep('config.dataset_configs.pattern_pathname', pattern_pathnames),
      hyper.sweep('config.dataset_configs.test_fold_num', test_fold_nums),
  ])
  init_from_domain = hyper.product([
      hyper.sweep('config.init_from.xm', wids),])

  return hyper.zipit([test_set_domain, init_from_domain])
