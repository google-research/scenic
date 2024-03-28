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
r"""Default configs for Plain ViT on ImageNet2012.

Based on: https://arxiv.org/pdf/2205.01580.pdf

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
  config.experiment_name = 'imagenet_vit'

  # Dataset.
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
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 50_000
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
  config.model.classifier = 'map'
  config.model.representation_size = None
  config.model.positional_embedding = 'learn'
  config.init_head_bias = -6.9  # -log(1000)

  # Pretrained model info.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_format = 'big_vision'
  config.init_from.checkpoint_path = CHECKPOINTS[VARIANT]

  # Training.
  config.trainer_name = 'plainvit_trainer'
  config.loss = 'sigmoid_xent'
  config.l2_decay_factor = None
  config.label_smoothing = None
  config.log_eval_steps = 1000
  config.log_summary_steps = 100
  config.num_training_epochs = None  # we use number of steps
  config.num_training_steps = 20_000
  config.rng_seed = 42
  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
  sched.lr_configs.steps_per_cycle = config.num_training_steps
  sched.lr_configs.total_steps = config.num_training_steps
  sched.lr_configs.warmup_steps = 500
  sched.lr_configs.base_learning_rate = 0.03

  # Configure both learning rate schedules.
  config.schedule = ml_collections.ConfigDict({'all': sched})

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scenic.momentum_hp'
  # Disable Optax gradient clipping as we handle it ourselves.
  optim.max_grad_norm = 1.0
  optim.weight_decay = 0.0
  config.optimizer = optim

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.2

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


def fixed(hyper, **kw):
  return hyper.zipit(
      [hyper.fixed(f'config.{k}', v, length=1) for k, v in kw.items()]
  )


def task(
    hyper, name, train, val, n_cls, steps, warmup, h_res=256, l_res=224, ch=3
):
  """Vision task with val and test splits."""
  # pylint: disable=line-too-long
  common = '|value_range(-1, 1)'
  common += f'|onehot({n_cls}, key="label", key_result="labels")'
  common += '|keep("image", "labels")'
  pp_train = f'decode(channels={ch})|inception_crop({l_res})|flip_lr' + common
  pp_eval = f'decode(channels={ch})|resize({l_res})' + common
  pp_eval_resize_crop = (
      f'decode(channels={ch})|resize({h_res})|central_crop({l_res})' + common
  )
  pp_eval_resmall_crop = (
      f'decode(channels={ch})|resize_small({h_res})|central_crop({l_res})'
      + common
  )
  # pylint: enable=line-too-long

  return fixed(
      hyper,
      **{
          'dataset_configs.dataset': name,
          'dataset_configs.train_split': train,
          'dataset_configs.pp_train': pp_train,
          'dataset_configs.val_split': (
              ('val', name, val, pp_eval),
              ('y/val_resize', name, val, pp_eval),
              ('y/val_resize_crop', name, val, pp_eval_resize_crop),
              ('y/val_resmall_crop', name, val, pp_eval_resmall_crop),
              ('test', name, 'test', pp_eval),
              ('y/test_resize', name, 'test', pp_eval),
              ('y/test_resize_crop', name, 'test', pp_eval_resize_crop),
              ('y/test_resmall_crop', name, 'test', pp_eval_resmall_crop),
          ),
          'dataset_configs.num_classes': n_cls,
          'schedule.all.lr_configs.warmup_steps': warmup,
          'schedule.all.lr_configs.total_steps': steps,
          'schedule.all.lr_configs.steps_per_cycle': steps,
          'num_training_steps': steps,
      },
  )


def get_hyper(hyper):
  """Sweeps over datasets."""
  # pylint: disable=line-too-long
  c100 = lambda **kw: task(hyper, 'cifar100', 'train[:98%%]', 'train[98%%:]', n_cls=100, **kw)
  c10 = lambda **kw: task(hyper, 'cifar10', 'train[:98%%]', 'train[98%%:]', n_cls=10, **kw)
  pet = lambda **kw: task(hyper, 'oxford_iiit_pet', 'train[:90%%]', 'train[90%%:]', n_cls=37, **kw)
  flower = lambda **kw: task(hyper, 'oxford_flowers102', 'train[:90%%]', 'train[90%%:]', n_cls=102, **kw)
  # pylint: enable=line-too-long

  return hyper.product([
      hyper.chainit([
          c100(h_res=448, l_res=384, steps=10_000, warmup=500),
          c10(h_res=448, l_res=384, steps=10_000, warmup=500),
          pet(h_res=448, l_res=384, steps=500, warmup=100),
          flower(h_res=448, l_res=384, steps=500, warmup=100),
      ]),
      hyper.sweep(
          'config.schedule.all.lr_configs.base_learning_rate',
          [0.03, 0.01, 0.003, 0.001],
      ),
  ])
