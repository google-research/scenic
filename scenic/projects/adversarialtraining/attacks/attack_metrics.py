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

"""Metrics for attack code."""
import functools

import jax.numpy as jnp
from scenic.projects.adversarialtraining.attacks import train_utils


def batchwise_scalar_to_metric_with_counts(metric, weights, devicewise_counts):
  norm_metric = train_utils.psum_metric_normalizer(
      (metric * weights, devicewise_counts))
  return norm_metric[0], devicewise_counts


def get_metrics(misc_artifacts, batch, metrics, images_to_log, metrics_fn):
  """Get the attack metrics."""
  logits = misc_artifacts['logits']
  adv_logits = misc_artifacts['adv_logits']
  adv_image = misc_artifacts['adv_image']
  adv_perturbation = misc_artifacts['adv_perturbation']

  adv_metrics = metrics_fn(adv_logits, batch)
  devicewise_counts = adv_metrics['loss'][1]

  batchwise_scalar_to_metric = functools.partial(
      batchwise_scalar_to_metric_with_counts,
      weights=batch.get('batch_mask'),
      devicewise_counts=devicewise_counts)

  # images to log
  images_to_log['adv_image'] = train_utils.unnormalize_imgnet(adv_image)
  images_to_log['adv_perturbation'] = train_utils.normalize_minmax(
      adv_perturbation)

  # simple train metrics
  metrics['steps_per_example'] = batchwise_scalar_to_metric(
      misc_artifacts['steps_per_example'])

  # simple network metrics
  acc_key = 'accuracy' if 'accuracy' in adv_metrics else 'prec@1'
  metrics['aux_accuracy'] = adv_metrics[acc_key]
  metrics['adv_loss'] = adv_metrics['loss']
  metrics['adv_loss-loss'] = metrics['adv_loss'][0] - metrics['loss'][
      0], devicewise_counts

  # adversarial interactions with the network
  l2_norms = jnp.sqrt(jnp.sum(adv_perturbation**2, axis=(1, 2, 3)))
  metrics['|adv-orig|_l2'] = batchwise_scalar_to_metric(l2_norms)

  # adversarial interactions with the network
  linfty_norms = jnp.max(jnp.abs(adv_perturbation), axis=(1, 2, 3))
  metrics['|adv-orig|_linfty'] = batchwise_scalar_to_metric(linfty_norms)
  metrics['|adv-orig|_stddev'] = batchwise_scalar_to_metric(
      jnp.std(jnp.abs(adv_perturbation), axis=(1, 2, 3)))

  logit_norms = jnp.sqrt(jnp.sum((adv_logits - logits)**2, axis=(1)))
  metrics['|adv_logits-logits|_l2'] = batchwise_scalar_to_metric(logit_norms)
  metrics[
      '|adv_logits-logits|_l2 / |adv-orig|_l2'] = batchwise_scalar_to_metric(
          logit_norms / (jnp.clip(l2_norms, a_min=1e-5)))

  # network performance on adversarial
  logits_are_correct = jnp.argmax(
      logits, axis=-1) == jnp.argmax(
          batch['label'], axis=-1)
  adv_logits_are_correct = jnp.argmax(
      adv_logits, axis=-1) == jnp.argmax(
          batch['label'], axis=-1)

  adv_correct_clean_incorrect = (adv_logits_are_correct >
                                 logits_are_correct).astype(jnp.float32)
  adv_incorrect_clean_correct = (adv_logits_are_correct <
                                 logits_are_correct).astype(jnp.float32)
  metrics['adv_correct_clean_incorrect'] = batchwise_scalar_to_metric(
      adv_correct_clean_incorrect)
  metrics['adv_incorrect_clean_correct'] = batchwise_scalar_to_metric(
      adv_incorrect_clean_correct)

  if 'target_labels_one_hot' in misc_artifacts:
    # We ran a targeted attack. This is to check that we hit the target after
    # running the attack.

    # This should be ~1/1000
    logits_are_target = jnp.argmax(
        logits, axis=-1) == jnp.argmax(
            misc_artifacts['target_labels_one_hot'], axis=-1)

    adv_logits_are_target = jnp.argmax(
        adv_logits, axis=-1) == jnp.argmax(
            misc_artifacts['target_labels_one_hot'], axis=-1)
    adv_are_target_clean_are_not_target = (
        adv_logits_are_target > logits_are_target).astype(jnp.float32)
    adv_are_not_target_clean_are_target = (
        adv_logits_are_target < logits_are_target).astype(jnp.float32)
    metrics['adv_are_target_clean_are_not_target'] = batchwise_scalar_to_metric(
        adv_are_target_clean_are_not_target)
    metrics['adv_are_not_target_clean_are_target'] = batchwise_scalar_to_metric(
        adv_are_not_target_clean_are_target)

  metrics['adv_loss_weight'] = misc_artifacts['adv_loss_weight'], 1

  return metrics, images_to_log
