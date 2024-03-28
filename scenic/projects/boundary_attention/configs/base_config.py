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

"""Base config for Boundary Attention."""

import datetime
from typing import Optional, Any

import ml_collections
from scenic.projects.boundary_attention.configs import dataset_configs
from scenic.projects.boundary_attention.configs import model_configs
from scenic.projects.boundary_attention.helpers import get_input_opts

 # Add path to the trained model here:
_CHECKPOINT_PATH = ''
_CHECKPOINT_STEP = -1
# If starting with pretrained weights, modify this instead
_MODEL_WEIGHTS_PATH = ''
# Add path to your data here:
_DATASET_DIR = ''
# Define to resize data to here (H, W, C) or set to None to use default size:
_INPUT_SIZE = None


def get_config(
    model_name: str = 'boundary_attention',
    checkpoint_path: Any = _CHECKPOINT_PATH,
    checkpoint_step: int = _CHECKPOINT_STEP,
    weights_path: Any = _MODEL_WEIGHTS_PATH,
    input_size: Optional[Any] = _INPUT_SIZE,
    dataset_name: str = 'kaleidoshapes',
    dataset_dir: Optional[str] = _DATASET_DIR,
    training_type: Optional[str] = 'train',
    runlocal: Optional[bool] = False,
    ) -> ml_collections.ConfigDict:
  """Returns base config for Boundary Attention."""

  runlocal = bool(runlocal)
  config = ml_collections.ConfigDict()
  config.model_name = model_name
  config.dataset_name = dataset_name
  config.training_type = training_type
  time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
  config.experiment_name = (model_name + '_' + dataset_name + '_' +
                            training_type + '_' + time_now)

  # Infra.
  # does eval on the train device
  config.eval_during_train = False
  config.disable_pmap_and_jit = False
  config.visualize = True

  # Dataset.
  config.dataset = dataset_configs.get_dataset_config(dataset_name,
                                                      dataset_dir,
                                                      input_size)
  config.batch_size = config.dataset.get('train_batchsize', 1)
  config.eval_batch_size = config.dataset.get('eval_batchsize', 1)

  # Model.
  config.model = model_configs.get_model_config(model_name)
  config.model.opts.training_type = training_type

  # Input Opts.
  config.model.input_opts = get_input_opts.get_input_opts(
      config.dataset.input_size, config.model.opts)

  # Initialize from.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = config.model.get('checkpoint_path',
                                                      checkpoint_path)
  config.init_from.checkpoint_step = config.model.get('checkpoint_step',
                                                      checkpoint_step)
  config.init_from.params_path = config.model.get('pretrained_params_path',
                                                  weights_path)

  # Training.
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.optimizer = 'adam'
  # config.optimizer_configs.weight_decay = 0.0
  config.optimizer_configs.skip_scale_and_bias_regularization = False
  config.l2_decay_factor = 0
  config.max_grad_norm = 10.0
  config.num_training_steps = int(300_000)
  config.log_eval_steps = 5000
  config.rng_seed = 42
  config.num_devices = None  # Updated by main()

  # Learning rate.
  base_lr = config.model.learning_rate
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * linear_decay * linear_warmup'
  config.lr_configs.warmup_steps = 5000
  config.lr_configs.total_steps = config.num_training_steps
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.base_learning_rate = base_lr
  config.lr_configs.end_learning_rate = base_lr / 10

  # Logging.
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps_per_device = 5_000  # used for backward compatibility
  config.checkpoint_steps = int(config.checkpoint_steps_per_device)
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval

  # Visualization config.
  config.viz_utils = ml_collections.ConfigDict()

  return config
