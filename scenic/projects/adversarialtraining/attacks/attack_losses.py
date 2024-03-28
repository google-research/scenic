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

"""Losses for attack."""
import jax
import jax.numpy as jnp


def get_attack_fn(attack_fn_str,
                  training_loss_fn_single,
                  batch,
                  train_state,
                  dropout_rng,
                  misc_artifacts,
                  attack_in_train_mode=True,
                  config=None):
  """Get specified attack function."""
  # create random label
  # ensure they don't change after initialization of loss
  true_labels = batch['label']
  random_labels = jax.random.randint(
      key=dropout_rng,
      shape=true_labels.shape[:-1],
      minval=0,
      maxval=config.advprop.num_classes,
      dtype=jnp.int16)

  # build loss functions
  def attack_fn_targeted(inputs, pertubation, target_labels):
    del pertubation

    new_batch = dict(batch)
    new_batch['inputs'] = inputs
    loss_clean, (_, _) = training_loss_fn_single(
        train_state.params,
        train_state.model_state,
        batch=new_batch,
        use_aux_batchnorm=True,
        use_aux_dropout=True,
        train_var=attack_in_train_mode,
    )

    target_labels_one_hot = jax.nn.one_hot(
        target_labels,
        num_classes=config.advprop.num_classes).astype(jnp.float32)

    new_batch['label'] = target_labels_one_hot
    misc_artifacts['target_labels_one_hot'] = target_labels_one_hot

    loss, (_, logits) = training_loss_fn_single(
        train_state.params,
        train_state.model_state,
        batch=new_batch,
        use_aux_batchnorm=True,
        use_aux_dropout=True,
        train_var=attack_in_train_mode,
    )

    loss_breakdown = {
        'loss_adv': loss,
        'loss_base': loss_clean,
    }
    return loss, (loss_breakdown, logits)

  def attack_fn_random_target(inputs, pertubation, labels=None):
    del labels
    return attack_fn_targeted(inputs, pertubation, target_labels=random_labels)

  # select loss function
  if attack_fn_str == 'random_target':
    full_attack_fn = attack_fn_random_target
  else:
    raise NotImplementedError('No implementation of %s' % attack_fn_str)

  return full_attack_fn, misc_artifacts
