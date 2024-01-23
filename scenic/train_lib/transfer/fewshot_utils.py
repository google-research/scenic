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

"""Utils for few-shot evaluation.

Adapted from:
https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/fewshot_lsr.py
"""

import functools
from typing import Any, Optional, Tuple

from absl import logging
from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib.big_transfer import builder as pp_builder
from scenic.train_lib import train_utils
import tensorflow_datasets as tfds

_BIAS_CONSTANT = 100.0
PyTree = Any


def prepare_data(xs, pad=None, num_devices=None):
  """Makes sure data fits local devices.

  Args:
    xs: the data to be reshaped (tree-map'able).
    pad: if not None, zero-pad arrays such that their first dimension has this
      size. Also introduces a "mask" key in `xs` that contains ones where the
      un-padded data resides and zeros where the padding is.
    num_devices: Number of devices to distribute the data across. If None,
      ``local_device_count`` will be used instead.

  Returns:
    `xs` but re-shaped to (local_devices, -1, ...) and optionally padded.
  """
  # Create a mask which will be 1 for real entries and 0 for padded ones.
  if pad is not None:
    xs['mask'] = np.full(len(xs['image']), 1.0)
  local_device_count = num_devices or jax.local_device_count()

  def _prepare(x):
    # Transforms x into read-only numpy array without copy if possible, see:
    # https://github.com/tensorflow/tensorflow/issues/33254#issuecomment-542379165
    x = np.asarray(memoryview(x))

    if pad is not None and len(x) != pad:
      # Append `pad - len(x)` rows of zeros.
      x = np.r_[x, np.zeros((pad - len(x),) + x.shape[1:], x.dtype)]

    return x.reshape((local_device_count, -1) + x.shape[1:])

  xs = jax.tree_util.tree_map(_prepare, xs)
  return {'inputs': xs['image'], 'label': xs['label'], 'batch_mask': xs['mask']}


def prepare_data_jit(xs, pad: Optional[int], global_devices: np.ndarray):
  """Prepare data-pipeline for jit-based backend."""

  if pad is not None:
    xs['mask'] = np.full(len(xs['image']), 1.0)

  def _pad(x):
    if pad is not None and len(x) != pad:
      x = np.asarray(memoryview(x))
      # Append `pad - len(x)` rows of zeros.
      x = np.r_[x, np.zeros((pad - len(x),) + x.shape[1:], x.dtype)]

    return x

  xs = jax.tree_util.tree_map(_pad, xs)
  xs = dataset_utils.shard_jit(xs, global_devices)
  return {'inputs': xs['image'], 'label': xs['label'], 'batch_mask': xs['mask']}


def start_input_pipeline(data, pad=None, num_devices=None, backend='pmap',
                         devices: Optional[np.ndarray] = None):
  train_iter = iter(data)
  if backend == 'pmap':
    train_iter = map(
        lambda x: prepare_data(x, pad, num_devices=num_devices), train_iter)
  elif backend == 'jit':
    train_iter = map(
        lambda x: prepare_data_jit(x, pad, global_devices=devices), train_iter)
  return train_iter


# Setup function for few-shot regression on CPU to avoid 'polluting' the TPU.
# It's fast enough when done in jax instead of numpy.
@functools.partial(jax.jit, backend='cpu', static_argnums=(5, 6))
def _fewshot_acc_fn(x: jnp.ndarray,
                    y: jnp.ndarray,
                    x_test: jnp.ndarray,
                    y_test: jnp.ndarray,
                    l2_reg: float,
                    num_classes: int,
                    target_is_one_hot: bool = False,
                    stddev_constant: float = 1e-5) -> float:
  """Computes (x,y) linear regression accuracy on (x_test, y_test).

  Args:
    x: An [n_examples, n_classes] matrix of feature representations. This will
      be whitened before computing linear regression.
    y: Array of labels. Shape is either [n_examples] or [n_examples, n_classes].
      In the latter case, target_is_one_hot must be True.
    x_test: An [n_test_examples, n_classes] matrix of feature representations.
      Will be whitened before computing linear regression.
    y_test: Array of labels. Shape is either [n_examples] or [n_examples,
      n_classes]. In the latter case, target_is_one_hot must be True.
    l2_reg: L2 regularisation co-efficient to apply when computing linear
      regression (also known as "ridge regression").
    num_classes: The number of classes in the dataset. Used to convert y to a
      one-hot representation if not already.
    target_is_one_hot: If the labels, y, are already one-hot or not.
    stddev_constant: Small constant to add when computing the standard deviation
      to avoid it being 0.

  Returns:
    The accuracy or precision@1 (for one-hot labels), after computing linear
      regression.
  """

  def preprocess_features(data: jnp.ndarray, mean: jnp.ndarray,
                          std: jnp.ndarray) -> jnp.ndarray:
    """Whitens features and adds a bias term."""
    data_whitened = (data - mean) / std
    # Add a constant feature for the bias, large so it's almost unregularized.
    data_whitened_bias = jnp.pad(
        data_whitened, ((0, 0), (0, 1)), constant_values=_BIAS_CONSTANT)
    return data_whitened_bias

  mean = jnp.mean(x, axis=0, keepdims=True)
  std = jnp.std(x, axis=0, keepdims=True) + stddev_constant

  x_whitened = preprocess_features(x, mean, std)
  x_test_whitened = preprocess_features(x_test, mean, std)

  # Solve linear regression problem.
  if not target_is_one_hot:
    y_one_hot = jax.nn.one_hot(y, num_classes)
  else:
    y_one_hot = y
  y_rescaled = 2.0 * y_one_hot - 1.0
  w = jnp.linalg.solve(
      x_whitened.T @ x_whitened + jnp.eye(x_whitened.shape[1]) * l2_reg,
      x_whitened.T @ y_rescaled)

  if target_is_one_hot:
    # Compute the precision@1 for multilabel datasets. This is the same as
    # accuracy if there is one active label.
    preds = x_test_whitened @ w
    top1_idx = jnp.argmax(preds, axis=-1)
    top1_correct = jnp.take_along_axis(y_test, top1_idx[..., None], axis=-1)
    top1_correct = jnp.squeeze(top1_correct)
    return jnp.mean(top1_correct)  # pytype: disable=bad-return-type  # jnp-type
  else:
    # Predict test-set values and measure their accuracy.
    preds = jnp.argmax(x_test_whitened @ w, axis=1)
    return jnp.mean(preds == y_test)  # pytype: disable=bad-return-type  # jnp-type


class FewShotEvaluator:
  """Class for few-shot evaluation."""

  def __init__(self, representation_fn, fewshot_config,
               backend: str = 'pmap', devices: Optional[jnp.ndarray] = None,
               out_shardings: Optional[PyTree] = None):
    self.shots = fewshot_config.shots
    self.l2_regs = fewshot_config.l2_regs
    self.local_batch_size = fewshot_config.batch_size // jax.process_count()
    self.pp_tr = fewshot_config.pp_train
    self.pp_te = fewshot_config.pp_eval
    self.walk_first = fewshot_config.walk_first
    self._datasets = {}  # This will be our cache for lazy loading.

    self.backend = backend
    assert self.backend in {
        'pmap',
        'jit',
    }, f'Unsupported backend: {self.backend}. Must be one of [pmap, jit].'
    if self.backend == 'jit':
      assert devices is not None, 'Devices must be provided when using jit.'
    self.devices = devices

    if self.backend == 'pmap':
      self.repr_fn = jax.pmap(
          representation_fn, donate_argnums=(1,), axis_name='batch'
      )
    elif self.backend == 'jit':
      self.repr_fn = jax.jit(
          representation_fn, donate_argnums=(1,), out_shardings=out_shardings
      )

  # Setup input pipeline.
  def _get_dataset(self, dataset, train_split, test_split):
    """Lazy-loads given dataset."""
    key = (dataset, train_split, test_split)
    try:
      return self._datasets[key]
    except KeyError:
      train_ds = dataset_utils.get_data(
          dataset=dataset,
          split=train_split,
          batch_size=self.local_batch_size,
          preprocess_fn=pp_builder.get_preprocess_fn(self.pp_tr),
          repeats=1,
          cache='loaded',
          drop_remainder=False)
      test_ds = dataset_utils.get_data(
          dataset=dataset,
          split=test_split,
          batch_size=self.local_batch_size,
          preprocess_fn=pp_builder.get_preprocess_fn(self.pp_te),
          repeats=1,
          cache='loaded',
          drop_remainder=False)
      num_classes = tfds.builder(dataset).info.features['label'].num_classes
      return self._datasets.setdefault(key, (train_ds, test_ds, num_classes))

  def _get_repr(self, train_state, data):
    """Compute representation for the whole dataset."""
    pre_logits_list = []
    labels_list = []
    for batch in start_input_pipeline(data,
                                      pad=self.local_batch_size,
                                      backend=self.backend,
                                      devices=self.devices):
      pre_logits, labels, mask = self.repr_fn(train_state, batch)
      # We need to unreplicate the output of `lax.all_gather`.
      # Shapes at this point are:
      #   pre_logits: `[hosts, devices, global_batch, features]`.
      #   labels: `[hosts, devices, global_batch]`.
      #   mask: `[hosts, devices, global_batch]`.
      if self.backend == 'pmap':
        pre_logits = jax_utils.unreplicate(pre_logits)

        if pre_logits.ndim != 3:
          raise ValueError('Shape of the representations sent to the linear '
                           'fewshot should be `[num_devices, bs, features]`.')

        mask = np.array(jax_utils.unreplicate(mask)).astype(bool)
        pre_logits_list.append(np.array(pre_logits)[mask])
        labels_list.append(np.array(jax_utils.unreplicate(labels))[mask])
      else:
        pre_logits = jax.device_get(pre_logits)

        if pre_logits.ndim != 2:
          raise ValueError('Shape of the representations sent to the linear '
                           'fewshot should be `[bs, features]`.')

        mask = np.array(jax.device_get(mask)).astype(bool)
        pre_logits_list.append(np.array(pre_logits)[mask])
        labels_list.append(np.array(jax.device_get(labels))[mask])

      if pre_logits.shape[-1] > 2048:
        logging.warning(
            'The feature size for the representations is too large'
            '(feature size = %d). This might cause severe slowdown '
            'of solving the linear equation.', pre_logits.shape[-1])

    pre_logits = np.concatenate(pre_logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return pre_logits, labels

  def compute_fewshot_metrics(self, train_state, dataset, train_split,
                              test_split):
    """Compute few-shot metrics on one dataset."""
    train_ds, test_ds, num_classes = self._get_dataset(dataset, train_split,
                                                       test_split)
    logging.info('[fewshot][%s]: Precomputing train (%s)', dataset, train_split)
    repr_train, labels_train = self._get_repr(train_state, train_ds)
    logging.info('[fewshot][%s]: Precomputing test (%s)', dataset, test_split)
    repr_test, labels_test = self._get_repr(train_state, test_ds)

    logging.info('[fewshot][%s]: solving systems', dataset)

    # Collect where we have samples of which classes.
    class_indices = [
        np.where(labels_train == cls_i)[0] for cls_i in range(num_classes)
    ]

    results = {}
    for shots in self.shots:
      all_idx = [indices[:shots] for indices in class_indices]
      all_idx = np.concatenate(all_idx, axis=0)
      x = repr_train[all_idx]
      y = labels_train[all_idx]

      for l2_reg in self.l2_regs:
        acc = _fewshot_acc_fn(x, y, repr_test, labels_test, l2_reg, num_classes)
        results[shots, l2_reg] = np.array(acc)
    return results

  def run_all(self, train_state, datasets):
    """Compute summary over all `datasets` that comes from config."""
    results = {}
    for name, dataset_args in datasets.items():
      results[name] = self.compute_fewshot_metrics(train_state, *dataset_args)

    # Now also figure out the regularization parameter that works best across
    # all datasets, per-shot. Similar to ATARI benchmark requiring one single
    # hyper-param across tasks, or BiT-HyperRule defining one clear thing.
    # Avoids over-fitting to a single task by selecting on test there, while
    # also avoiding the need to do cross-validation runs for each task.
    best_l2 = {}
    for shots in self.shots:
      reg_ranks = []
      for name, res in results.items():
        reg_accus = [res[shots, l2] for l2 in self.l2_regs]
        reg_ranks.append(np.argsort(np.argsort(reg_accus)))
      best_l2[shots] = self.l2_regs[np.argmax(np.mean(reg_ranks, axis=0))]

    return results, best_l2

  def log_fewshot_summary(self, writer: metric_writers.MetricWriter, step,
                          results):
    """Call `writer` with a descriptive string and the results."""
    results, best_l2 = results
    scalars = {}

    # First, go through each individual result:
    for dataset_name, result in results.items():
      for (shots, l2), acc in result.items():
        scalars[f'zz/{dataset_name}_{shots}shot_l2={l2}'] = acc

    # Second, report each dataset/shot with the single 'globally' best l2.
    for shots, l2 in best_l2.items():
      scalars[f'z/best_l2_for_{shots}shot_image_eval'] = l2

      for dataset_name, result in results.items():
        scalars[f'z/{dataset_name}_{shots}shot'] = result[shots, l2]

    # And a highlight, if desired:
    if self.walk_first:
      dataset_name, shots = self.walk_first
      l2 = best_l2[shots]
      highlight_value = results[dataset_name][shots, l2]
      scalars[f'a/{dataset_name}_{shots}shot'] = highlight_value

    writer.write_scalars(step, scalars)


def min_without_none(a: Optional[int], b: Optional[int]):
  """Returns the minimum of two integers, ignoring None values."""
  if a is None and b is None:
    raise ValueError('At least one argument should not be None')
  if a is None:
    return b
  if b is None:
    return a
  return min(a, b)


class FewShotEvaluatorVideo:
  """Class for few-shot evaluation."""

  def __init__(self, representation_fn, fewshot_config):
    self.config = fewshot_config
    self.shots = fewshot_config.shots
    self.l2_regs = fewshot_config.l2_regs
    self.local_batch_size = fewshot_config.batch_size // jax.process_count()
    self.repr_fn = jax.pmap(
        representation_fn, donate_argnums=(1,), axis_name='batch')
    self.walk_first = fewshot_config.get('walk_first')
    self._datasets = {}  # This will be our cache for lazy loading.

  def _get_dataset(self,
                   dataset_name: str,
                   train_split: str,
                   test_split: str,
                   num_train_examples: Optional[int] = None,
                   num_test_examples: Optional[int] = None):
    """Lazy-loads given dataset."""
    assert train_split == 'train', ('train_split should be set to train for '
                                    'few-shot-video evaluator')
    assert (test_split == 'validation' or test_split == 'test'), (
        'test_split should be set to validation or test for few-shot-video '
        'evaluator')
    if dataset_name in self._datasets:
      return self._datasets[dataset_name]
    else:
      rng = jax.random.PRNGKey(self.config.rng_seed)
      data_rng, rng = jax.random.split(rng)
      with self.config.unlocked():
        self.config.dataset_name = dataset_name
      dataset = train_utils.get_dataset(self.config, data_rng)
      train_ds = dataset.train_iter
      num_train_samples = min_without_none(
          num_train_examples, dataset.meta_data['num_train_examples'])
      if test_split == 'validation':
        test_ds = dataset.valid_iter
        num_test_samples = min_without_none(
            num_test_examples, dataset.meta_data['num_eval_examples'])
      elif test_split == 'test':
        test_ds = dataset.test_iter
        num_test_samples = min_without_none(
            num_test_examples, dataset.meta_data['num_test_examples'])
      num_classes = dataset.meta_data['num_classes']
      is_one_hot = dataset.meta_data['target_is_onehot']
      return self._datasets.setdefault(
          dataset_name, (train_ds, test_ds, num_train_samples, num_test_samples,
                         num_classes, is_one_hot))

  def _get_repr(self, train_state, data, num_samples):
    """Compute representation for the whole dataset."""
    pre_logits_list = []
    labels_list = []
    total_steps = int(np.ceil(num_samples / self.config.batch_size))
    for _ in range(1, total_steps + 1):
      batch = next(data)
      pre_logits, labels, mask = self.repr_fn(train_state, batch)
      # We need to unreplicate the output of `lax.all_gather`.
      # Shapes at this point are:
      #   pre_logits: `[hosts, devices, global_batch, features]`.
      #   labels: `[hosts, devices, global_batch]`.
      #   mask: `[hosts, devices, global_batch]`.
      pre_logits = jax_utils.unreplicate(pre_logits)
      if pre_logits.ndim != 3:
        raise ValueError('Shape of the representations sent to the linear '
                         'fewshot should be `[num_devices, bs, features]`.')
      if pre_logits.shape[-1] > 2048:
        logging.warning(
            'The feature size for the representations is too large'
            '(feature size = %d). This might cause severe slowdown '
            'of solving the linear equation.', pre_logits.shape[-1])
      mask = np.array(jax_utils.unreplicate(mask)).astype(bool)
      pre_logits_list.append(np.array(pre_logits)[mask])
      labels_list.append(np.array(jax_utils.unreplicate(labels))[mask])
    pre_logits = np.concatenate(pre_logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return pre_logits, labels

  def compute_fewshot_metrics(self, train_state, dataset, train_split,
                              test_split):
    """Compute few-shot metrics on one dataset."""

    (train_ds, test_ds, num_train_samples, num_test_samples, num_classes,
     is_one_hot) = self._get_dataset(dataset, train_split, test_split,
                                     self.config.get('num_train_examples'),
                                     self.config.get('num_test_examples'))
    logging.info('[fewshot][%s]: Precomputing train (%s)', dataset, train_split)
    repr_train, labels_train = self._get_repr(train_state, train_ds,
                                              num_train_samples)
    labels_train = jnp.squeeze(labels_train)
    logging.info('[fewshot][%s]: Precomputing test (%s)', dataset, test_split)
    repr_test, labels_test = self._get_repr(train_state, test_ds,
                                            num_test_samples)
    labels_test = jnp.squeeze(labels_test)
    logging.info('[fewshot][%s]: Solving linear system', dataset)
    # Collect where we have samples of which classes.
    if is_one_hot:
      class_indices = [
          np.where(labels_train[:, cls_i] == 1)[0]
          for cls_i in range(num_classes)
      ]
    else:
      class_indices = [
          np.where(labels_train == cls_i)[0] for cls_i in range(num_classes)
      ]

    results = {}
    for shots in self.shots:
      all_idx = [indices[:shots] for indices in class_indices]
      all_idx = np.concatenate(all_idx, axis=0)
      x = repr_train[all_idx]
      y = labels_train[all_idx]
      for l2_reg in self.l2_regs:
        acc = _fewshot_acc_fn(x, y, repr_test, labels_test, l2_reg, num_classes,
                              is_one_hot)
        results[shots, l2_reg] = np.array(acc)
    return results

  def run_all(self, train_state, datasets):
    """Compute summary over all `datasets` that comes from config."""
    results = {}
    for name, dataset_args in datasets.items():
      results[name] = self.compute_fewshot_metrics(train_state, *dataset_args)

    # Now also figure out the regularization parameter that works best across
    # all datasets, per-shot. Similar to ATARI benchmark requiring one single
    # hyper-param across tasks, or BiT-HyperRule defining one clear thing.
    # Avoids over-fitting to a single task by selecting on test there, while
    # also avoiding the need to do cross-validation runs for each task.
    best_l2 = {}
    for shots in self.shots:
      reg_ranks = []
      for name, res in results.items():
        reg_accuracies = [res[shots, l2] for l2 in self.l2_regs]
        reg_ranks.append(np.argsort(np.argsort(reg_accuracies)))
      best_l2[shots] = self.l2_regs[np.argmax(np.mean(reg_ranks, axis=0))]

    return results, best_l2

  def log_fewshot_summary(self,
                          writer: metric_writers.MetricWriter,
                          step: int,
                          results: Tuple[train_utils.PyTree,
                                         train_utils.PyTree],
                          prefix_detailed: str = 'zz_fewshot_detailed',
                          prefix_best_l2: str = 'fewshot',
                          prefix_highlight: str = 'fewshot_main',
                          flush_writer: bool = True):
    """Call `writer` with a descriptive string and the results."""
    results, best_l2 = results
    scalars = {}

    # First, go through each individual result:
    for dataset_name, result in results.items():
      for (shots, l2), acc in result.items():
        scalars[f'{prefix_detailed}/{dataset_name}_{shots}shot_l2={l2}'] = acc

    # Second, report each dataset/shot with the single 'globally' best l2.
    for shots, l2 in best_l2.items():
      scalars[f'{prefix_detailed}/best_l2_for_{shots}shot_video_eval'] = l2

      for dataset_name, result in results.items():
        value = result[shots, l2]
        scalars[f'{prefix_best_l2}/{dataset_name}_{shots}shot'] = value

    # And a highlight, if desired:
    if self.walk_first:
      dataset_name, shots = self.walk_first
      l2 = best_l2[shots]
      value = results[dataset_name][shots, l2]
      scalars[f'{prefix_highlight}/{dataset_name}_{shots}shot'] = value

    writer.write_scalars(step, scalars)
    if flush_writer:
      writer.flush()
