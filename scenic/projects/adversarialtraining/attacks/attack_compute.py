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

"""Run the adversarial attacks."""
import functools

import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.adversarialtraining.attacks import attack_losses
from scenic.projects.adversarialtraining.attacks import attack_methods
from scenic.projects.adversarialtraining.attacks import attack_transforms
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers


def typecheck(obj):
  return hasattr(obj, '__iter__') and not isinstance(obj, str)


def get_adv_pyramid(batch,
                    clean_logits,
                    attack_fn,
                    optimizer_def,
                    config,
                    dropout_rng,
                    lr_rng,
                    misc_artifacts):
  """Get pyramid attack for batch."""
  del clean_logits, dropout_rng

  epsilon = config.advprop.epsilon
  num_steps = config.advprop.num_steps
  # 1 is global, 224 is pixel-level
  # pylint: disable=eval-used
  pyramid_sizes = eval(config.advprop.pyramid_sizes)
  pyramid_scalars = eval(config.advprop.pyramid_scalars)
  if not typecheck(pyramid_sizes):
    pyramid_sizes = [pyramid_sizes]
  if not typecheck(pyramid_scalars):
    pyramid_scalars = [pyramid_scalars]

  local_batch_size = batch['inputs'].shape[0]
  init_mode = config.advprop.get('init_mode', default=None)
  init_aug_params = {}
  if init_mode == 'zero':
    for (layer, pyramid_size) in enumerate(pyramid_sizes):
      init_aug_params[str(layer)] = jnp.zeros(
          dtype=jnp.float32,
          shape=(local_batch_size, pyramid_size, pyramid_size, 3))
  elif init_mode == 'random':
    for (layer, pyramid_size) in enumerate(pyramid_sizes):
      init_aug_params[str(layer)] = jax.random.uniform(
          key=lr_rng,
          dtype=jnp.float32,
          shape=(local_batch_size, pyramid_size, pyramid_size, 3),
          maxval=epsilon,
          minval=-epsilon)
  elif init_mode == 'normal':
    for (layer, pyramid_size) in enumerate(pyramid_sizes):
      init_aug_params[str(layer)] = epsilon * jax.random.normal(
          key=lr_rng,
          dtype=jnp.float32,
          shape=(local_batch_size, pyramid_size, pyramid_size, 3))
  else:
    raise NotImplementedError('Not valid init_mode %s' % init_mode)

  init_aug_scalars = {}
  if pyramid_scalars is None:
    for (layer, pyramid_size) in enumerate(pyramid_sizes):
      init_aug_scalars[str(layer)] = jnp.ones((len(pyramid_sizes)))
  else:
    for (layer, pyramid_size) in enumerate(pyramid_sizes):
      init_aug_scalars[str(
          layer)] = pyramid_scalars[layer] * jnp.ones((len(pyramid_sizes),))

  transform_fn = functools.partial(
      attack_transforms.patched_color_jitter,
      aug_fn=attack_transforms.fast_color_perturb)
  def transform_fn_pyramid(input_image, aug_params_dict):
    for layer in range(len(aug_params_dict)):
      input_image = transform_fn(
          input_image,
          init_aug_scalars[str(layer)] * aug_params_dict[str(layer)])
    return input_image

  adv_image, adv_perturbation, _, attack_artifacts = attack_methods.pgd_attack_transform(
      loss_fn=attack_fn,
      transform_fn=transform_fn_pyramid,
      init_aug_params=init_aug_params,
      input_image=batch['inputs'],
      label=batch['label'],
      epsilon=epsilon,
      num_steps=num_steps,
      rng=lr_rng,
      optimizer_def=optimizer_def,
      projection=attack_methods.project_perturbation_pyramid_inf,
  )
  local_result_advprop_pyramid = adv_image, adv_perturbation
  for key in attack_artifacts:
    misc_artifacts[key] = attack_artifacts[key]

  return (lambda _: local_result_advprop_pyramid), misc_artifacts


def get_adversarial_fn(adversarial_fn_name):
  if adversarial_fn_name == 'advprop_pyramid':
    return get_adv_pyramid
  else:
    raise NotImplementedError('No implementation for %s' % adversarial_fn_name)


def get_optimizer_def(optimizer_str, learning_rate_fn):
  """Get optimizer for adversarial attack."""
  optimizer_config = ml_collections.ConfigDict()
  if optimizer_str == 'GradientDescent':
    optimizer_config.optimizer = 'sgd'
  elif optimizer_str == 'Adam':
    optimizer_config.optimizer = 'adam'
    optimizer_config.b1 = 0.5
    optimizer_config.b2 = 0.5
  elif optimizer_str == 'AdaBelief':
    optimizer_config.optimizer = 'adabelief'
    optimizer_config.b1 = 0.5
    optimizer_config.b2 = 0.5
  else:
    raise NotImplementedError('advprop.optimizer is not valid: %s' %
                              optimizer_str)
  return functools.partial(
      optimizers.get_optimizer,
      optimizer_config=optimizer_config,
      learning_rate_fn=learning_rate_fn)


def get_adversarial_image_and_perturbation(batch, clean_logits, config,
                                           training_loss_fn_single, train_state,
                                           dropout_rng, lr_rng):
  """Get adversarial image and perturbation."""
  # initialize misc_artifacts which is piped through loss and attack
  misc_artifacts = {}

  # get adversarial modes
  if not config.get('adversarial_augmentation_mode'):
    raise NotImplementedError('Adversarial should receive a mode')
  adversarial_augmentation_mode = config.get('adversarial_augmentation_mode')
  if ',' in adversarial_augmentation_mode:
    adversarial_augmentation_modes = list(
        config.get('adversarial_augmentation_mode').split(',')
    )
  elif isinstance(adversarial_augmentation_mode, str):
    adversarial_augmentation_modes = [config.get(
        'adversarial_augmentation_mode')]
  else:
    raise NotImplementedError('adversarial_augmentation_mode is not valid: %s' %
                              str(adversarial_augmentation_mode))
  adversarial_fns = [get_adversarial_fn(adversarial_name)
                     for adversarial_name in adversarial_augmentation_modes]

  # get loss function
  attack_in_train_mode = config.advprop.get('attack_in_train_mode',
                                            default=True)
  attack_fn_str = config.advprop.get('attack_fn_str', default='random_target')
  attack_fn, misc_artifacts = attack_losses.get_attack_fn(
      attack_fn_str, training_loss_fn_single, batch, train_state, dropout_rng,
      misc_artifacts, attack_in_train_mode, config)

  # get optimizer def
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config.advprop)
  optimizer_str = config.advprop.get('optimizer', default='GradientDescent')
  optimizer_def = get_optimizer_def(optimizer_str, learning_rate_fn)

  # initialize augmentation params
  images_and_perturbations = []

  # run all augmentations
  for adv_fn in adversarial_fns:
    ims_and_pert, misc_artifacts = adv_fn(
        batch, clean_logits, attack_fn, optimizer_def, config, dropout_rng,
        lr_rng, misc_artifacts)
    images_and_perturbations.append(ims_and_pert)

  # sanity check
  assert len(images_and_perturbations) == len(adversarial_augmentation_modes)

  # randomly run one of them
  random_index = jax.random.randint(
      key=dropout_rng,
      shape=(),
      minval=0,
      maxval=len(adversarial_augmentation_modes))
  adversarial_image, adversarial_perturbation = jax.lax.switch(
      random_index, images_and_perturbations, None)

  return adversarial_image, adversarial_perturbation, misc_artifacts
