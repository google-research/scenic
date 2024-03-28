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
r"""Default configs for TopVit on abnormality baselines.

"""
# pylint: disable=line-too-long

import ml_collections

# TODO(shamsiz) Make a test dataset and replace abnormality, and job_type with
# dummy variables.

# Constants
_TOTAL_STEPS = 2_000
_NUM_TRAIN_NORMALS = {
    'del5_simple': 6089,
    'del5_net': 6089,
    't922_chrm22': 3721,
    't922_chrm9': 3670,
}
_NUM_TRAIN_ABNORMALS = {
    'del5_simple': 1056,
    'del5_net': 1386,
    't922_chrm22': 515,
    't922_chrm9': 387,
}
_TRAINER_NAME = {
    'start_from_scratch': 'classification_trainer',
    'finetune': 'transfer_trainer',
    'eval': 'inference',
}
_DEFAULT_JOB_TYPE = 'finetune'
_DEFAULT_ABNORMALITY = 'del5_simple'


def get_config(runlocal='',
               job_type=_DEFAULT_JOB_TYPE,
               abnormality=_DEFAULT_ABNORMALITY):
  """Returns the TopViT config for abnormality baseline task."""
  runlocal = bool(runlocal)
  if abnormality not in [
      'del5_simple', 'del5_net', 't922_chrm22', 't922_chrm9'
  ]:
    raise ValueError('abnormality must be specified; got "%r"' % abnormality)
  if job_type not in ['start_from_scratch', 'finetune', 'eval']:
    raise ValueError('job_type must be specified; got "%r"' % job_type)
  num_train_normals = _NUM_TRAIN_NORMALS[abnormality]
  num_train_abnormals = _NUM_TRAIN_ABNORMALS[abnormality]
  trainer_name = _TRAINER_NAME[job_type]

  config = ml_collections.ConfigDict()
  config.experiment_name = '%s-topvit-%s' % (abnormality, job_type)
  # Dataset.
  config.dataset_name = 'abnormality_baseline'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.abnormality = abnormality
  # For available cropped shapes, see abnormality_baseline_dataset:DATASET_PREFIXES.
  config.dataset_configs.chrm_image_shape = (199, 99)
  config.dataset_configs.replica = 0
  config.dataset_configs.num_abnormal = num_train_abnormals
  config.dataset_configs.num_normal = num_train_normals

  # Model.
  config.model_name = 'topological_vit_classification'
  config.model = ml_collections.ConfigDict()
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.0
  config.model_dtype_str = 'float32'
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [4, 4]
  config.model.hidden_size = 768
  config.model.num_heads = 12
  config.model.mlp_dim = 768
  config.model.num_layers = 16
  if job_type != 'start_from_scratch':
    # Pretrained model info.
    config.init_from = ml_collections.ConfigDict()
    config.init_from.xm = (36063788, 15)  # ChrmID topvit model
    config.init_from.checkpoint_path = None

  # Training.
  config.trainer_name = trainer_name
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
  config.batch_size = 8 if runlocal else 512
  config.rng_seed = 42
  config.init_head_bias = -10.0
  config.class_balancing = False
  config.save_predictions = True  # Save predictions in eval mode.

  # Learning rate.
  if job_type == 'start_from_scratch':
    base_lr = 3e-3
  else:
    base_lr = 3e-5
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = _TOTAL_STEPS
  config.lr_configs.end_learning_rate = 1e-6
  config.lr_configs.warmup_steps = 1_000
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


def get_hyper(hyper,
              job_type=_DEFAULT_JOB_TYPE,
              abnormality=_DEFAULT_ABNORMALITY):
  """Defines the hyper-parameters sweeps for doing grid search."""
  if abnormality not in [
      'del5_simple', 'del5_net', 't922_chrm22', 't922_chrm9'
  ]:
    raise ValueError('abnormality must be specified; got "%r"' % abnormality)
  if job_type not in ['start_from_scratch', 'finetune', 'eval']:
    raise ValueError('job_type must be specified; got "%r"' % job_type)
  num_train_abnormals = _NUM_TRAIN_ABNORMALS[abnormality]

  if job_type == 'eval':
    # Model inference will be run for each (xid, wid) in the list below.
    if abnormality == 'del5_simple':
      hparams = [
          hyper.sweep(
              'config.init_from.xm',
              [
                  (42902421, 25),  # from-scratch
                  (42902394, 25),  # fine-tuning
              ])
      ]
    elif abnormality == 'del5_net':
      hparams = [hyper.sweep('config.init_from.xm', [])]  # ChrmID model is default
    elif abnormality == 't922_chrm22':
      hparams = [
          hyper.sweep(
              'config.init_from.xm',
              [
                  (43256082, 25),  # fine-tuning
                  (43275768, 25),  # from-scratch
              ])
      ]
    elif abnormality == 't922_chrm9':
      hparams = [
          hyper.sweep(
              'config.init_from.xm',
              [
                  (43275868, 25),  # fine-tuning
                  (43252546, 25),  # from-scratch
              ])
      ]
  else:
    hparams = [
        hyper.chainit([
            hyper.product([
                hyper.sweep('config.dataset_configs.num_abnormal',
                            [3, 10, 30, 300]),
                hyper.sweep('config.dataset_configs.replica', [0, 1, 2, 3, 4]),
            ]),
            hyper.product([
                hyper.sweep('config.dataset_configs.num_abnormal',
                            [num_train_abnormals]),
                hyper.sweep('config.dataset_configs.replica', [0, 0, 0, 0, 0]),
            ]),
        ])
    ]
  return hyper.chainit(hparams)
