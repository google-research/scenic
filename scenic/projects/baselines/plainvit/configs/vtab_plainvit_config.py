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
r"""Default configs for Plain ViT on VTAB.

Based on: https://arxiv.org/pdf/2106.04560.pdf

ViT-G/14 gets 78.29 +/- 0.53 on VTAB (see https://arxiv.org/pdf/2106.04560.pdf).

"""
# pylint: disable=line-too-long

import ml_collections

VARIANT = 'B/16'

HIDDEN_SIZES = {
    'Ti': 192,
    'S': 384,
    'M': 512,
    'B': 768,
    'L': 1024,
    'H': 1280,
    'g': 1408,
    'G': 1664,
    'e': 1792
}
MLP_DIMS = {
    'Ti': 768,
    'S': 1536,
    'M': 2048,
    'B': 3072,
    'L': 4096,
    'H': 5120,
    'g': 6144,
    'G': 8192,
    'e': 15360
}
NUM_HEADS = {
    'Ti': 3,
    'S': 6,
    'M': 8,
    'B': 12,
    'L': 16,
    'H': 16,
    'g': 16,
    'G': 16,
    'e': 16
}
NUM_LAYERS = {
    'Ti': 12,
    'S': 12,
    'M': 12,
    'B': 12,
    'L': 24,
    'H': 32,
    'g': 40,
    'G': 48,
    'e': 56
}

CHECKPOINTS = {
    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    'S/16': 'gs://big_vision/vit_s16_i1k_300ep.npz',
}


def get_config():
  """Returns the ViT experiment configuration for ImageNet."""

  config = ml_collections.ConfigDict()
  config.experiment_name = 'vtab_vit_g'

  # Dataset (changed in the sweep).
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = ''
  config.dataset_configs.val_split = ()
  config.dataset_configs.train_split = ''
  config.dataset_configs.num_classes = 0
  config.dataset_configs.pp_train = ''
  config.dataset_configs.pp_eval = None
  config.dataset_configs.prefetch_to_device = None
  config.dataset_configs.shuffle_buffer_size = 1_000
  config.batch_size = 512

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'plainvit'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = HIDDEN_SIZES[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = NUM_HEADS[version]
  config.model.mlp_dim = MLP_DIMS[version]
  config.model.num_layers = NUM_LAYERS[version]
  config.model.dropout_rate = 0.
  config.model.classifier = 'gap'  # ViT-G was tuned on VTAB using 'gap'.
  config.model.representation_size = None
  config.model.positional_embedding = 'learn'
  config.init_head_bias = 0.0

  # Pretrained model info.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_format = 'big_vision'
  config.init_from.checkpoint_path = CHECKPOINTS[VARIANT]

  # Training.
  config.trainer_name = 'plainvit_trainer'
  config.loss = 'softmax_xent'
  config.l2_decay_factor = None
  config.label_smoothing = None
  config.log_eval_steps = 100
  config.log_summary_steps = 50
  config.num_training_epochs = None  # we use number of steps
  config.num_training_steps = 2_500  # ViT-G was tuned on VTAB with 2500 steps.
  config.rng_seed = 42
  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
  sched.lr_configs.steps_per_cycle = config.num_training_steps
  sched.lr_configs.total_steps = config.num_training_steps
  sched.lr_configs.warmup_steps = 200  # ViT-G was tuned on VTAB with 200 warmup steps.
  sched.lr_configs.base_learning_rate = 0.01  # ViT-G was tuned on VTAB with 0.01.

  # Configure both learning rate schedules.
  config.schedule = ml_collections.ConfigDict({'all': sched})

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scenic.momentum_hp'
  # Disable Optax gradient clipping as we handle it ourselves.
  optim.max_grad_norm = 1.0
  optim.weight_decay = 0.0
  config.optimizer = optim

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


def fix1(hyper, key, val):
  return hyper.fixed(f'config.{key}', val, length=1)


def task(hyper,
         name,
         train,
         test,
         n_cls,
         steps=None,
         warmup=None,
         lr=None,
         ch=3,
         base_pp='',
         label='label',
         crop=True,
         flip=True,
         h_res=256,
         l_res=224):
  """Vision task with val and test splits."""
  common = '|value_range(-1, 1)'
  common += f'|onehot({n_cls},key="{label}",key_result="labels")'
  common += '|keep("image", "labels")'
  pp_train = f'decode(channels={ch})|{base_pp}'
  if crop:
    pp_train += f'resize({h_res})|random_crop({l_res})'
  else:
    pp_train += f'resize({l_res})'
  if flip:
    pp_train += '|flip_lr'
  pp_train += common
  pp_eval = f'decode(channels={ch})|{base_pp}resize({l_res})' + common
  pp_eval_resize_crop = f'decode(channels={ch})|{base_pp}resize({h_res})|central_crop({l_res})' + common
  pp_eval_resmall_crop = f'decode(channels={ch})|{base_pp}resize_small({h_res})|central_crop({l_res})' + common

  task_info = [
      fix1(hyper, 'dataset_configs.dataset', name),
      fix1(hyper, 'dataset_configs.train_split', train),
      fix1(hyper, 'dataset_configs.val_split', (
          ('test', name, test, pp_eval),
          ('y/test_resize', name, test, pp_eval),
          ('y/test_resize_crop', name, test, pp_eval_resize_crop),
          ('y/test_resmall_crop', name, test, pp_eval_resmall_crop),
      )),
      fix1(hyper, 'dataset_configs.num_classes', n_cls),
      fix1(hyper, 'dataset_configs.pp_train', pp_train),
  ]
  schedule = []
  if steps:
    schedule += [fix1(hyper, 'schedule.all.lr_configs.total_steps', steps)]
    schedule += [fix1(hyper, 'schedule.all.lr_configs.steps_per_cycle', steps)]
    schedule += [fix1(hyper, 'num_training_steps', steps)]
  if warmup:
    schedule += [fix1(hyper, 'schedule.all.lr_configs.warmup_steps', warmup)]
  if lr:
    schedule += [fix1(hyper, 'schedule.all.lr_configs.base_learning_rate', lr)]

  return hyper.zipit(task_info + schedule)


def vtab_b_tasks(hyper):
  """Vision task with val and test splits."""
  # Note: steps, warmup and learning rate were tuned using vtab_ft_val.py.
  tasks = hyper.chainit([
      # Resize, crop, flip
      task(
          hyper,
          'caltech101:3.*.*',
          'train[:800]+train[2754:2954]',
          'test',
          102,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'diabetic_retinopathy_detection/btgraham-300:3.*.*',
          'train[:800]+validation[:200]',
          'test',
          5,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'dtd:3.*.*',
          'train[:800]+validation[:200]',
          'test',
          47,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'oxford_flowers102:2.*.*',
          'train[:800]+validation[:200]',
          'test',
          102,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'oxford_iiit_pet:3.*.*',
          'train[:800]+train[2944:3144]',
          'test',
          37,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'resisc45:3.*.*',
          'train[:800]+train[18900:19100]',
          'train[25200:]',
          45,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'sun397/tfds:4.*.*',
          'train[:800]+validation[:200]',
          'test',
          397,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'cifar100:3.*.*',
          'train[:800]+train[45000:45200]',
          'test',
          100,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'eurosat/rgb:2.*.*',
          'train[:800]+train[16200:16400]',
          'train[21600:]',
          10,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'patch_camelyon:2.*.*',
          'train[:800]+validation[:200]',
          'test',
          2,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'smallnorb:2.*.*',
          'train[:800]+test[:200]',
          'test[50%%:]',
          9,
          label='label_elevation',
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'svhn_cropped:3.*.*',
          'train[:800]+train[65931:66131]',
          'test',
          10,
          crop=True,
          flip=True,
          h_res=448,
          l_res=384),

      # Resize, crop
      task(
          hyper,
          'dsprites:2.*.*',
          'train[:800]+train[589824:590024]',
          'train[663552:]',
          16,
          base_pp='dsprites_pp("label_orientation",16)|',
          crop=True,
          flip=False,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'smallnorb:2.*.*',
          'train[:800]+test[:200]',
          'test[50%%:]',
          18,
          label='label_azimuth',
          crop=True,
          flip=False,
          h_res=448,
          l_res=384),

      # Resize, flip
      task(
          hyper,
          'clevr:3.*.*',
          'train[:800]+train[63000:63200]',
          'validation',
          6,
          base_pp='clevr_pp("closest_object_distance")|',
          crop=False,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'clevr:3.*.*',
          'train[:1000]+train[63000:63200]',
          'validation',
          8,
          base_pp='clevr_pp("count_all")|',
          crop=False,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'dmlab:2.0.1',
          'train[:800]+validation[:200]',
          'test',
          6,
          crop=False,
          flip=True,
          h_res=448,
          l_res=384),
      task(
          hyper,
          'kitti:3.1.0',
          'train[:800]+validation[:200]',
          'test',
          4,
          base_pp='kitti_pp("closest_vehicle_distance")|',
          crop=False,
          flip=True,
          h_res=448,
          l_res=384),

      # Resize
      task(
          hyper,
          'dsprites:2.*.*',
          'train[:800]+train[589824:590024]',
          'train[663552:]',
          16,
          base_pp='dsprites_pp("label_x_position",16)|',
          crop=False,
          flip=False,
          h_res=448,
          l_res=384),
  ])
  return tasks


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([
      hyper.sweep('config.rng_seed', [0, 1, 2]),
      vtab_b_tasks(hyper),
  ])
