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

"""Mixin training config."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin_hmdb51
from scenic.projects.lang4video.configs.datasets import mixin_imagenet
from scenic.projects.lang4video.configs.datasets import mixin_kinetics400
from scenic.projects.lang4video.configs.datasets import mixin_kinetics600
from scenic.projects.lang4video.configs.datasets import mixin_kinetics700
from scenic.projects.lang4video.configs.datasets import mixin_msr_vtt
from scenic.projects.lang4video.configs.datasets import mixin_ucf101
from scenic.projects.lang4video.configs.datasets import mixin_youcook2


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  run_local_str = run_local

  config = ml_collections.ConfigDict()

  config.trainer_name = 'visual_text_trainer'

  # config.num_training_epochs = 32
  config.num_training_steps = 100000

  config.optimizer = 'radamw'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.weight_decay = 0.01
  config.optimizer_configs.skip_scale_and_bias_regularization = True
  config.optimizer_configs.gradient_centralization = False
  config.optimizer_configs.lookahead = False
  config.optimizer_configs.lookahead_sync_period = 5
  config.optimizer_configs.lookahead_slow_step_size = 0.5
  config.optimizer_configs.params_to_freeze = ()
  config.max_grad_norm = 0.5
  config.label_smoothing = None  # TODO(sacastro): unused.
  config.temperature = 0.01
  config.fit_temperature = False  # TODO(sacastro): unused.

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'linear_warmup * constant * cosine_decay'
  config.lr_configs.warmup_steps = .1
  config.lr_configs.steps_per_cycle = 1.
  config.lr_configs.base_learning_rate = 5e-4

  config.log_summary_steps = 20
  # config.log_eval_steps = 100
  config.eval_while_training = False
  config.checkpoint_steps = 200

  # The following inherit configs from CLIP, but we're going to use them in a
  # way that we won't use the model's properties.
  config.evaluation_configs = (
      mixin_hmdb51.get_config(run_local_str),
      mixin_imagenet.get_config(run_local_str),
      mixin_kinetics400.get_config(run_local_str),
      mixin_kinetics600.get_config(run_local_str),
      mixin_kinetics700.get_config(run_local_str),
      mixin_msr_vtt.get_config(run_local_str),
      mixin_ucf101.get_config(run_local_str),
      mixin_youcook2.get_config(run_local_str),
  )

  return config
