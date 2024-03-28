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
r"""A config for training layout_ denoise model.

"""
import copy
import ml_collections
from scenic.projects.layout_denoise.configs import dataset_config


def get_config():
  """Config for training on ui_layout with layout_vit."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'LayoutDenoise'
  shuffle_buffer_size = 5_000
  config.datasets = {}
  config.dataset_names = [
      'rico',
  ]
  config.total_steps = 15_000
  config.dataset_weight_stages = [config.total_steps]
  config.dataset_weights = [[
      1,
  ] * len(config.dataset_names)]
  config.use_inner = True
  # PADDING + BACKGROUND + 24 layout classes.
  config.num_classes = 26
  for d_name in config.dataset_names:
    config.datasets[d_name] = dataset_config.get_config(
        d_name,
        shuffle_buffer_size=shuffle_buffer_size // len(config.dataset_names),
        use_inner=config.use_inner)

  # Model.
  config.model_name = 'layout_denoise'
  # model_type can be `full`, `vh_only` or `mlp`.
  config.model_type = 'full'
  config.binary_task = False
  config.binary_label_weight = 5.0
  config.model_dtype_str = 'float32'
  config.hidden_dim = 256
  config.vocab_size = 28_536
  # Add path to the vocab here:
   config.vocab_path = ('')
  config.grid_size = 32
  config.grid_rows = 34
  config.grid_cols = 34
  config.num_masks = 10
  config.image_range = config.grid_rows * config.grid_cols
  config.max_num_boxes = 128
  config.max_image_size = 1080
  config.modal_ranges = [
      config.image_range, config.image_range + config.max_num_boxes
  ]
  config.query_emb_size = None  # Same as hidden_size.
  config.transformer_num_heads = 8
  config.transformer_num_encoder_layers = 6
  config.transformer_num_decoder_layers = 6
  config.transformer_qkv_dim = 256
  config.transformer_mlp_dim = 2048
  config.transformer_normalize_before = False
  config.backbone_num_filters = 64
  config.backbone_num_layers = 50
  config.class_dropout_rate = .0
  config.dropout_rate = 0.2
  config.attention_dropout_rate = 0.2
  config.pos_pattern = '1/4'

  # Loss.
  config.aux_loss = True
  config.class_loss_coef = 1.0
  # Training.
  config.trainer_name = 'layout_denoise_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.weight_decay = 1e-4
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.max_grad_norm = 0.1
  config.label_smoothing = 0.1
  config.num_training_epochs = 200
  config.batch_size = 128
  config.eval_batch_size = 32
  config.rng_seed = 0
  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*piecewise_constant'
  config.lr_configs.decay_events = [
      5_000,
  ]
  # note: this is absolute (not relative):
  config.lr_configs.decay_factors = [.1]
  config.lr_configs.base_learning_rate = 1e-4

  # backbone traiing configs: optimizer and learning rate
  config.backbone_training = ml_collections.ConfigDict()
  config.backbone_training.optimizer = copy.deepcopy(config.optimizer)
  config.backbone_training.optimizer_configs = copy.deepcopy(
      config.optimizer_configs)
  config.backbone_training.lr_configs = copy.deepcopy(config.lr_configs)
  config.backbone_training.lr_configs.base_learning_rate = 6e-5
  config.l2_decay_factor = .000001

  # pretrained_backbone
  # TODO(dehghani): use pretrain_utils and clean up this part
  config.load_pretrained_backbone = True
  config.freeze_backbone_batch_stats = True
  config.pretrained_backbone_configs = ml_collections.ConfigDict()
  config.pretrained_backbone_configs.xm = (18140063, 1)
  config.pretrained_backbone_configs.checkpoint_path = None
  # Logging.
  config.eval_ngram_list = [1]
  config.run_coco_evaluation = True  # Run evaluation using.
  config.write_summary = True
  config.xprof = False  # Profile using xprof.
  config.log_summary_steps = 100  # train summary steps
  config.log_large_summary_steps = 1000
  config.log_eval_steps = 1000
  config.checkpoint_steps = 1000
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.eval_synchronously = False
  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
