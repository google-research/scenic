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
r"""Default configs for COCO detection using DeformableDETR.
"""
# pylint: enable=line-too-long

import ml_collections

_COCO_TRAIN_SIZE = 118287
batch_size = ml_collections.FieldReference(32)
num_epochs = ml_collections.FieldReference(50)
steps_per_epoch = _COCO_TRAIN_SIZE // batch_size


def get_coco_config():
  """Returns the configuration for COCO detection using DeformableDETR."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'coco_detection_deformable_detr'

  # Compiler
  config.compiler_config = ml_collections.ConfigDict()
  config.compiler_config.train_remat = True
  config.compiler_config.attention_batching_mode = 'auto'

  # Dataset.
  config.dataset_name = 'coco_deformable_detr_detection'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 10_000
  config.dataset_configs.max_boxes = 299
  config.dataset_configs.max_size = 1333
  config.dataset_configs.valid_max_size = 1333
  config.data_dtype_str = 'float32'

  # Model.
  config.model_dtype_str = 'float32'
  config.model_name = 'deformable_detr'
  config.matcher = 'hungarian'
  config.num_classes = 91
  config.embed_dim = 256
  config.enc_embed_dim = 256
  config.num_queries = 300
  config.num_feature_levels = 4
  config.num_heads = 8
  config.num_encoder_layers = 6
  config.num_decoder_layers = 6
  config.transformer_ffn_dim = 1024
  config.num_enc_points = 4
  config.num_dec_points = 4
  config.backbone_num_filters = 64
  config.backbone_num_layers = 50
  config.dropout_rate = 0.1

  # Loss.
  config.aux_loss = True
  config.bbox_loss_coef = 5.0
  config.giou_loss_coef = 2.0
  config.class_loss_coef = 2.0
  config.focal_loss_alpha = 0.25
  config.focal_loss_gamma = 2.0
  # Use the mean num_boxes as normalization.
  config.normalization = 'global'

  # Training.
  config.trainer_name = 'deformable_detr_trainer'
  config.num_training_epochs = num_epochs
  config.rng_seed = 0
  config.batch_size = batch_size
  config.eval_batch_size = batch_size * 2

  # Optimization.
  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.weight_decay = 1e-4
  config.optimizer_config.beta1 = 0.9
  config.optimizer_config.beta2 = 0.999
  config.optimizer_config.base_learning_rate = 2e-4
  config.optimizer_config.max_grad_norm = 0.1
  config.optimizer_config.learning_rate_decay_rate = 0.1
  config.optimizer_config.learning_rate_reduction = 0.1
  config.optimizer_config.learning_rate_decay_event = (
      num_epochs * 4 // 5 * steps_per_epoch)

  # Pretrained_backbone.
  config.load_pretrained_backbone = True
  config.freeze_backbone_batch_stats = True
  config.pretrained_backbone_configs = ml_collections.ConfigDict()

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 200  # Train summary steps.
  # Expensive summary operations freq.
  config.log_large_summary_steps = steps_per_epoch.identity()
  # Train steps before eval, typically one epoch.
  config.log_eval_steps = steps_per_epoch.identity()
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = steps_per_epoch.identity()
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


