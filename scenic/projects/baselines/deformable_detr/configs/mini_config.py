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

r"""Mini config for COCO detection using DeformableDETR.
"""

from scenic.projects.baselines.deformable_detr.configs.common import get_coco_config


def get_config():
  """Returns the configuration for COCO using a mini DeformableDETR."""
  config = get_coco_config()

  # Dataset.
  config.dataset_configs.max_boxes = 10

  # Model.
  config.embed_dim = 32
  config.enc_embed_dim = 32
  config.num_queries = 12
  config.num_feature_levels = 4
  config.num_heads = 4
  config.num_encoder_layers = 2
  config.num_decoder_layers = 2
  config.transformer_ffn_dim = 256
  config.num_enc_points = 1
  config.num_dec_points = 2
  config.backbone_num_filters = 16
  config.backbone_num_layers = 18

  # Pretrained_backbone.
  config.load_pretrained_backbone = False
  config.freeze_backbone_batch_stats = False

  # Logging.
  config.write_summary = False  # don't write summary
  config.checkpoint = False  # don't do checkpointing
  config.checkpoint_steps = None
  config.debug_train = False  # don't debug  during training
  config.debug_eval = False  # don't debug during eval

  config.num_training_steps = 2
  config.log_eval_steps = 2
  config.steps_per_eval = 2
  config.num_training_epochs = None

  return config
