# Copyright 2025 The Scenic Authors.
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
r"""Default configs for ViT on ImageNet2012.



Note: you can also use ImageNet input pipeline from big transfer pipeline:
```
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  # aka tiny_test/test[:5%] in task_adapt
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.num_classes = 1000
  INPUT_RES = 224  # pylint: disable=invalid-name
  RESIZE_RES = int(INPUT_RES * (256 / 224))  # pylint: disable=invalid-name
  LS = 1e-4  # pylint: disable=invalid-name
  config.dataset_configs.pp_train = (
    f'decode_jpeg_and_inception_crop({INPUT_RES})|flip_lr|value_range(-1, '
    f'1)|onehot({config.dataset_configs.num_classes},'
    f' key="label", key_result="labels", '
    f'on={1.0-LS}, off={LS})|keep("image", '
    f'"labels")')  # pylint: disable=line-too-long
  config.dataset_configs.pp_eval = (
    f'decode|resize_small({RESIZE_RES})|'
    f'central_crop({INPUT_RES})|value_range(-1, '
    f'1)|onehot({config.dataset_configs.num_classes},'
    f' key="label", '
    f'key_result="labels")|keep("image", '
    f'"labels")')  # pylint: disable=line-too-long
  config.dataset_configs.prefetch_to_device = 2

  # shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

```

"""
# pylint: disable=line-too-long

import ml_collections

UCF101_TRAIN_SIZE = 30779
UCF101_VAL_SIZE = 11137
UCF101_TEST_SIZE = 11137

# _IMAGENET_TRAIN_SIZE = 1281167
# VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'vivit_k400_classification-vit'
  # Dataset.
  config.dataset_name = 'video_tfrecord_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()

  config.dataset_configs.base_dir = (
      "/users/undergraduate/rjchen25/k400/tfrecord_0301")
  config.dataset_configs.tables = {
      'train': 'filted_train_set@175',
      'validation': 'filted_test_set@105',
      'test': 'filted_test_set@105'
  }
  config.dataset_configs.examples_per_subset = {
      'train': UCF101_TRAIN_SIZE,
      'validation': UCF101_VAL_SIZE,
      'test': UCF101_TEST_SIZE
  }
  config.dataset_configs.num_classes = 400
  config.data_dtype_str = 'float32'

  # This is going to sample 32 frames, sampled at a stride of 2 from the video.
  # Kinetics videos has 250 frames.
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 2
  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  """
  # Multicrop evaluation settings:
  #config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.do_multicrop_test = False  
  config.dataset_configs.log_test_epochs = 5
  # The effective batch size per host when testing is
  # num_test_clips * test_batch_size.
  config.dataset_configs.num_test_clips = 4
  #config.dataset_configs.test_batch_size = 1  # Must equal num_local_devices.
  config.dataset_configs.test_batch_size = 1
  # To take three spatial crops when testing.
  config.dataset_configs.do_three_spatial_crops = True
  config.multicrop_clips_per_device = 2
  """

  # Model.

  config.model_name = 'vivit_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = 768

  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'factorized_encoder'

  config.pretrain_checkpoint = ml_collections.ConfigDict()
  config.pretrain_checkpoint.model_config = None
  config.pretrain_checkpoint.checkpoint_path = "/users/undergraduate/rjchen25/scenic-vivit/checkpoint"

  config.model.patches = ml_collections.ConfigDict()
  config.model.spatial_transformer = ml_collections.ConfigDict()
  config.model.spatial_transformer.num_heads = 12
  config.model.spatial_transformer.mlp_dim = 3072
  config.model.spatial_transformer.num_layers = 12
  config.model.temporal_transformer = ml_collections.ConfigDict()
  config.model.temporal_transformer.num_heads = 12
  config.model.temporal_transformer.mlp_dim = 3072
  config.model.temporal_transformer.num_layers = 12
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.1
  config.model.dropout_rate = 0.1
  config.model_dtype_str = 'float32'
  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.patches.size = (16, 16, 2)

  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  # Applies when temporal_encoding_config.method='temporal_sampling'
  config.model.temporal_encoding_config.n_sampled_frames = 16  # Unused here.

  # Training.
  config.trainer_name = 'ricky_trainer'
  config.optimizer = 'adamw'
  # fixing optax
  #config.optimizer = 'adam'


  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.weight_decay = 1e-4   
  config.l2_decay_factor = 0
  #config.max_grad_norm = 1
  config.max_grad_norm = 0.5
  config.label_smoothing = None
  config.num_training_epochs = 80
  #config.batch_size = 64
  config.batch_size = 2
  config.rng_seed = 0

  # Use ImageNet-21k-initialized model.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_config = None
  # Download pretrained ImageNet checkpoints from here:
  # https://github.com/google-research/scenic/tree/main/scenic/projects/baselines (checkpoint_format = 'scenic')  pylint: disable=line-too-long
  # https://github.com/google-research/vision_transformer (checkpoint_format = 'big_vision')  pylint: disable=line-too-long
  config.init_from.checkpoint_path = '/users/undergraduate/rjchen25/test_scenic/scenic/scenic/projects/vivit/checkpoints/test0/'
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.model_config = ml_collections.ConfigDict()
  config.init_from.model_config.model = ml_collections.ConfigDict()
  config.init_from.model_config.model.classifier = 'token'  # Specify if this is 'token' or 'gap'.  pylint: disable=line-too-long
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'tile'

  # Learning rate.
  accumulation_steps = 8
  steps_per_epoch = UCF101_TRAIN_SIZE // config.batch_size
  #total_steps = config.num_training_epochs * steps_per_epoch
  total_micro_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  #config.lr_configs.warmup_steps = int(2.5 * steps_per_epoch)
  config.lr_configs.warmup_steps = int(2.5 * steps_per_epoch * accumulation_steps)
  #config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.steps_per_cycle = total_micro_steps 
  #config.lr_configs.base_learning_rate = 3e-4
  config.lr_configs.base_learning_rate = 8e-5
  #config.log_summary_steps = 2

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = True  # Debug mode during training.
  config.debug_eval = True  # Debug mode during eval.
  config.checkpoint_steps = 1000  # Checkpoint more frequently than a val epoch.
  return config