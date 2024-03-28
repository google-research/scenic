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
r"""Configs for COCO detection using DETR with Sinkhorn as matching algorithm.

"""
# pylint: enable=line-too-long

import copy
import ml_collections
_COCO_TRAIN_SIZE = 118287
NUM_EPOCHS = 300


def get_config():
  """Returns the configuration for COCO detection using DETR."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'coco_detection_detr'

  # Dataset.
  config.dataset_name = 'coco_detr_detection'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 10_000
  # Should be `config.num_queries - 1` because (i) Sinkhorn currently requires
  # square cost matrices; and (ii) an additional empty box is appended inside
  # the model.
  config.dataset_configs.max_boxes = 99
  config.data_dtype_str = 'float32'

  # Model.
  config.model_dtype_str = 'float32'
  config.model_name = 'detr'
  config.matcher = 'sinkhorn'
  config.hidden_dim = 256
  config.num_queries = 100
  config.query_emb_size = None  # Same as hidden_size.
  config.transformer_num_heads = 8
  config.transformer_num_encoder_layers = 6
  config.transformer_num_decoder_layers = 6
  config.transformer_qkv_dim = 256
  config.transformer_mlp_dim = 2048
  config.transformer_normalize_before = False
  config.backbone_num_filters = 64
  config.backbone_num_layers = 50
  config.dropout_rate = 0.
  config.attention_dropout_rate = 0.1

  # Sinkhorn.
  # See https://ott-jax.readthedocs.io/en/latest/notebooks/One_Sinkhorn.html
  # for more insights about the meanings and effects of those parameters.
  config.sinkhorn_epsilon = 1e-3
  # Speeds up convergence using epsilon decay. Start with a value 50 times
  # higher than the target and decay by a factor 0.9 between iterations.
  config.sinkhorn_init = 50
  config.sinkhorn_decay = 0.9
  config.sinkhorn_num_iters = 1000  # Sinkhorn number of iterations.
  config.sinkhorn_threshold = 1e-2  # Reconstruction threshold.
  # Starts using momemtum after after 100 Sinkhorn iterations.
  config.sinkhorn_chg_momentum_from = 100
  config.sinkhorn_num_permutations = 100

  # Loss.
  config.aux_loss = True
  config.bbox_loss_coef = 5.0
  config.giou_loss_coef = 2.0
  config.class_loss_coef = 1.0
  config.eos_coef = 0.1

  # Training.
  config.trainer_name = 'detr_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.weight_decay = 1e-4
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.max_grad_norm = 0.1
  config.num_training_epochs = NUM_EPOCHS
  config.batch_size = 64
  config.rng_seed = 0

  decay_events = {500: 400}

  # Learning rate.
  steps_per_epoch = _COCO_TRAIN_SIZE // config.batch_size
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*piecewise_constant'
  config.lr_configs.decay_events = [
      decay_events.get(NUM_EPOCHS, NUM_EPOCHS * 2 // 3) * steps_per_epoch,
  ]
  # Note: this is absolute (not relative):
  config.lr_configs.decay_factors = [.1]
  config.lr_configs.base_learning_rate = 1e-4

  # Backbone training configs: optimizer and learning rate.
  config.backbone_training = ml_collections.ConfigDict()
  config.backbone_training.optimizer = copy.deepcopy(config.optimizer)
  config.backbone_training.optimizer_configs = copy.deepcopy(
      config.optimizer_configs)
  config.backbone_training.lr_configs = copy.deepcopy(config.lr_configs)
  config.backbone_training.lr_configs.base_learning_rate = 1e-5

  # Pretrained_backbone.
  config.load_pretrained_backbone = True
  config.freeze_backbone_batch_stats = True
  config.pretrained_backbone_configs = ml_collections.ConfigDict()
  # Download pretrained ResNet50 checkpoints from here:
  # https://github.com/google-research/scenic/tree/main/scenic/projects/baselines pylint: disable=line-too-long
  config.init_from.checkpoint_path = 'path_to_checkpoint_of_resnet_50'

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 50  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = steps_per_epoch
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


