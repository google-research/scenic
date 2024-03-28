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
r"""Default configs for AdaTape on ImageNet.


"""
# pylint: enable=line-too-long

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000
VARIANT = 'S/16'

HIDDEN_SIZE = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}
NUM_HEADS = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}
MLP_DIM = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}
NUM_LAYERS = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}


def get_config(runlocal=''):
  """Returns the AdaTape_ViT experiment configuration for JFT."""
  config = ml_collections.ConfigDict()

  # Dataset.
  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-regularized_adavit'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr'
      '|randaug(2, 15)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'adatape'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = HIDDEN_SIZE[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = NUM_HEADS[version]
  config.model.mlp_dim = MLP_DIM[version]
  config.model.num_layers = NUM_LAYERS[version]
  config.model.classifier = 'gap'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  config.model.ac_config = ml_collections.ConfigDict()
  config.model.ac_config.add_tape_token_to_layers = (1,)
  # 'from_input', 'random_learnable', 'from_tape_bank'
  config.model.ac_config.num_tape_tokens = 10
  config.model.ac_config.tape_bank_size = 10000
  config.model.ac_config.tt_mlp_dim = MLP_DIM[version]
  config.model.ac_config.tt_dropout_rate = 0.0
  config.model.ac_config.enc_tt_mlp_dim = 0
  config.model.ac_config.enc_tt_dropout_rate = 0.0
  config.model.ac_config.split_tt = NUM_HEADS[version]
  config.model.ac_config.bank_norm = True
  config.model.ac_config.bank_type = 'input'  # learn
  # Two following options will be activated only when bank_type == input.
  config.model.ac_config.patch_bank_size = 8
  config.model.ac_config.query_type = config.model.classifier

  # Dynamic length.
  config.model.ac_config.dynamic_tape_length = None
  config.model.ac_config.dynamic_tape_length = ml_collections.ConfigDict()
  config.model.ac_config.dynamic_tape_length.num_token_per_step = 1
  config.model.ac_config.dynamic_tape_length.act_epsilon = 2.0
  config.model.ac_config.dynamic_tape_length.act_loss_weight = 0.01
  config.model.ac_config.dynamic_tape_length.act_loss_type = 'entropy'
  # max, entropy
  # These two types works comparable in our experiments
  config.model.ac_config.dynamic_tape_length.bernoulli_p = 0.0
  config.model.ac_config.dynamic_tape_length.complex_query = True
  config.model.ac_config.dynamic_tape_length.query_noise = 0.0

  # Training.
  config.trainer_name = 'adatape_classify_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 42
  config.init_head_bias = -6.9  # -log(1000)

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.5

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  config.m = None  # Placeholder for randaug strength.
  config.l = None  # Placeholder for randaug layers.


  return config


