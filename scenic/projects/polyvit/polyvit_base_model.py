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

"""Base class for models working with polyvit."""

import functools
from typing import Any, Dict, Optional, Tuple, List

from absl import logging
from flax.training import common_utils
from immutabledict import immutabledict
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.classification_model import classification_metrics_function
from scenic.model_lib.base_models.encoder_decoder_model import encoder_decoder_metrics_function
from scenic.model_lib.base_models.multilabel_classification_model import multilabel_classification_metrics_function
from scenic.projects.vivit import model_utils as vivit_model_utils


class Task:
  """Defines name of all possible tasks."""
  BOW = 'bow'
  SEQ = 'seq'
  LABEL = 'label'
  MULTILABEL = 'multilabel'
  MULTIHEADLABEL = 'multiheadlabel'
  FEWSHOT = 'fewshot'  # This is a special task for a fewshot bottleneck.


class Modality:
  """Defines name of all possible modalities."""
  IMAGE = 'image'
  VIDEO = 'video'
  AUDIO = 'audio'  # Represented as a spectrogram.


_BOW_CLASSIFICATION_METRICS = immutabledict({
    'prec@1': (model_utils.weighted_top_one_correctly_classified,
               model_utils.num_examples),
    'loss': (model_utils.weighted_unnormalized_sigmoid_cross_entropy,
             model_utils.num_examples)
})

_MULTIHEADLABEL_METRICS = immutabledict({
    'accuracy': (model_utils.weighted_correctly_classified,
                 model_utils.num_examples),
    'accuracy_top_5': (functools.partial(
        model_utils.weighted_topk_correctly_classified,
        k=5), model_utils.num_examples),
    'loss': (model_utils.weighted_unnormalized_softmax_cross_entropy,
             model_utils.num_examples)
})


def bow_classification_metrics_function(
    logits: jnp.ndarray,
    batch: base_model.Batch,
    target_is_multihot: bool = False,
    metrics: base_model.MetricNormalizerFnDict = _BOW_CLASSIFICATION_METRICS,
) -> Dict[str, Tuple[float, int]]:
  """Calcualte metrics for the Bag of Words classification task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_multihot: If the target is a multi-hot vector.
   metrics: The multi-label classification metrics to evaluate. The key is the
     name of the  metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_multihot:
    multihot_target = batch['label']
  else:
    # This is to support running a multi-label classification model on
    # single-label classification tasks:
    multihot_target = common_utils.onehot(batch['label'], logits.shape[-1])

  # multihot_target is initially one-hot of shape (bs, len, vocab_size),
  # while we actually need multi-hot of shape (bs, vocab_size).
  multihot_target = multihot_target.max(axis=-2)

  weights = batch.get('batch_mask')  # batch_mask might not be defined

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(  # pytype: disable=wrong-arg-types  # jax-ndarray
        (val[0](logits, multihot_target, weights),  # pytype: disable=wrong-arg-types  # jax-types
         val[1](logits, multihot_target, weights)))  # pytype: disable=wrong-arg-types  # jax-types
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-types


def multihead_classification_metrics_function(
    logits,
    batch,
    metrics: base_model.MetricNormalizerFnDict = _MULTIHEADLABEL_METRICS,
    class_splits: Optional[jnp.ndarray] = None,
    split_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Returns a callable metric function for the multihead classification task.

  Args:
    logits: Output of model in shape [batch, length, num_classes].
    batch: Batch of data that has 'label' and optionally 'batch_mask'.
    metrics: The multi-label classification metrics to evaluate. The key is the
      name of the  metric, and the value is the metrics function.
    class_splits: start indices of class splits.
    split_names: names of class splits.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """

  one_hot_targets = batch['label']
  weights = batch.get('batch_mask')  # batch_mask might not be defined

  logit_splits = jnp.split(logits, class_splits, axis=-1)[:-1]
  one_hot_target_splits = jnp.split(
      one_hot_targets, class_splits, axis=-1)[:-1]

  evaluated_metrics = {}
  total_loss = [0.0, 0]
  for logits_i, one_hot_targets_i, name in zip(logit_splits,
                                               one_hot_target_splits,
                                               split_names):
    for key, val in metrics.items():
      evaluated_metrics[
          f'{name}_{key}'] = model_utils.psum_metric_normalizer(  # pytype: disable=wrong-arg-types  # jax-ndarray
              (val[0](logits_i, one_hot_targets_i,
                      weights), val[1](logits_i, one_hot_targets_i,
                                       weights)))
      if key == 'loss':
        total_loss[0] += evaluated_metrics[f'{name}_{key}'][0]
        total_loss[1] += evaluated_metrics[f'{name}_{key}'][1]
  evaluated_metrics['total_loss'] = tuple(total_loss)

  if len(class_splits) == 2:
    pairwise_acc = model_utils.psum_metric_normalizer(
        (vivit_model_utils.joint_accuracy(logits, one_hot_targets, class_splits,
                                          weights),
         model_utils.num_examples(logits, one_hot_targets, weights)))
    pairwise_top_five = model_utils.psum_metric_normalizer(
        (vivit_model_utils.joint_top_k(
            logits, one_hot_targets, class_splits, k=5, weights=weights),
         model_utils.num_examples(logits, one_hot_targets, weights)))
    eval_name = f'{split_names[0]}-{split_names[1]}'
    evaluated_metrics[f'{eval_name}_accuracy'] = pairwise_acc
    evaluated_metrics[f'{eval_name}_accuracy_top_5'] = pairwise_top_five

  return evaluated_metrics


def classification_metrics_function_with_acc_top_5(*args, **kwargs):
  """A wrapper over classification_metrics_function which has accuracy_top_5."""
  return classification_metrics_function(
            *args, metrics=_MULTIHEADLABEL_METRICS, **kwargs)


_METRICS_FUNCTIONS = {
    Task.BOW:
        bow_classification_metrics_function,
    Task.SEQ:
        encoder_decoder_metrics_function,
    Task.LABEL: classification_metrics_function_with_acc_top_5,
    Task.MULTILABEL:
        multilabel_classification_metrics_function,
    Task.MULTIHEADLABEL:
        multihead_classification_metrics_function
}


def polyvit_metrics_function(
    logits: jnp.ndarray,
    batch: base_model.Batch,
    dataset_name: str,
    dataset_meta_data: Dict[str, Any],
    class_splits: Dict[str, Any],
    split_names: Dict[str, Any]
) -> Dict[str, Tuple[float, int]]:
  """Defines and computes metrics for the polyvit model.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   dataset_name: The name of the dataset used for the task.
   dataset_meta_data: Metadata of the dataset we are using.
   class_splits: start indices of class splits for multi-head classification
     tasks.
   split_names: names of class splits for multi-head classification tasks.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """

  current_dataset_meta_data = dataset_meta_data[dataset_name]
  task = current_dataset_meta_data['task']

  kwargs = {}

  if task in [Task.MULTILABEL, Task.BOW]:
    kwargs = {
        'target_is_multihot':
            current_dataset_meta_data.get('target_is_onehot', False)
    }
  elif task == Task.MULTIHEADLABEL:
    kwargs = {
        'class_splits': class_splits[dataset_name],
        'split_names': split_names[dataset_name]
    }
  else:
    kwargs = {
        'target_is_onehot':
            current_dataset_meta_data.get('target_is_onehot', False)
    }

  return _METRICS_FUNCTIONS[task](logits, batch, **kwargs)  # pytype: disable=wrong-keyword-args


def compute_multihead_label_loss(logits: jnp.ndarray,
                                 one_hot_targets: jnp.ndarray,
                                 weights: Optional[jnp.ndarray],
                                 class_splits: Optional[jnp.ndarray],
                                 label_smoothing: Optional[float] = None):
  """Computes loss for the multi-head label task.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   one_hot_targets: ground truth labels of the same shape as logits.
   weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
   class_splits: start indices of class splits.
   label_smoothing: float scalar to use to smooth the one-hot labels.

  Returns:
    Loss value.
  """

  if logits.shape[-1] != class_splits[-1]:
    raise AssertionError('Logit dimension must be equal to number of classes')

  logit_splits = jnp.split(logits, class_splits, axis=-1)[:-1]
  one_hot_target_splits = jnp.split(
      one_hot_targets, class_splits, axis=-1)[:-1]

  sof_ce_losses = [
      model_utils.weighted_softmax_cross_entropy(
          logits, one_hot_targets, weights, label_smoothing)
      for logits, one_hot_targets in zip(logit_splits, one_hot_target_splits)
  ]
  sof_ce_loss = jnp.mean(jnp.array(sof_ce_losses))

  return sof_ce_loss


_LOSS_FUNCTIONS = {Task.BOW: model_utils.weighted_sigmoid_cross_entropy,
                   Task.SEQ: model_utils.weighted_softmax_cross_entropy,
                   Task.LABEL: model_utils.weighted_softmax_cross_entropy,
                   Task.MULTILABEL: model_utils.weighted_sigmoid_cross_entropy,
                   Task.MULTIHEADLABEL: compute_multihead_label_loss}


class PolyVitBaseModel(base_model.BaseModel):
  """PolyVit base model."""

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Dict[str, Any]]) -> None:
    if config is None:
      logging.warning('You are creating the model with default config.')
      config = self.default_flax_model_config()
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.flax_model = self.build_flax_model()

  def _get_splits(self):
    """Returns class_splits and split_names."""

    class_splits = {}
    split_names = {}

    for ds_name, cfg in self.config.datasets.items():
      # The first condition is needed for disabling datasets in hyperparameter
      # sweeps.
      if ds_name in self.dataset_meta_data and self.dataset_meta_data[ds_name][
          'task'] == Task.MULTIHEADLABEL:
        assert cfg.get('class_splits'), ('class_splits must be specified')
        class_splits[ds_name] = np.cumsum(cfg.class_splits)
        if cfg.get('split_names'):
          split_names[ds_name] = cfg.split_names
        else:
          split_names[ds_name] = [
              str(x + 1) for x in range(len(class_splits[ds_name]))
          ]

    return class_splits, split_names

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    class_splits, split_names = self._get_splits()
    return functools.partial(
        polyvit_metrics_function, dataset_meta_data=self.dataset_meta_data,
        class_splits=class_splits, split_names=split_names)

  def loss_function(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    dataset_name: str,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      dataset_name: The name of the dataset used for the task.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')
    class_splits, _ = self._get_splits()

    current_dataset_meta_data = self.dataset_meta_data[dataset_name]
    task = current_dataset_meta_data['task']

    if current_dataset_meta_data.get(
        'target_is_onehot', False):
      one_or_multi_hot_targets = batch['label']
    elif task == Task.MULTIHEADLABEL:
      raise ValueError('Target labels should be one-hot.')
    else:
      one_or_multi_hot_targets = common_utils.onehot(batch['label'],
                                                     logits.shape[-1])

    if self.config.get('label_smoothing_params') is not None:
      label_smoothing = self.config.label_smoothing_params.get(dataset_name,
                                                               None)
    else:
      label_smoothing = self.config.get('label_smoothing')

    kwargs = {'label_smoothing': label_smoothing}

    if task == Task.MULTIHEADLABEL:
      kwargs['class_splits'] = class_splits[dataset_name]

    if task == Task.BOW:
      # one_or_multi_hot_targets is initially one-hot of shape (bs, len,
      # vocab_size), while we actually need multi-hot of shape (bs, vocab_size).
      one_or_multi_hot_targets = one_or_multi_hot_targets.max(axis=-2)

    loss = _LOSS_FUNCTIONS[task](
        logits,
        one_or_multi_hot_targets,
        weights,
        **kwargs)

    config_loss_weight = self.config.get(
        'loss_weights', ml_collections.ConfigDict())
    loss_weight = config_loss_weight.get(dataset_name, 1.0)

    loss = loss * loss_weight

    if self.config.get('l2_decay_factor') is None:
      total_loss = loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jnp-type

  def build_flax_model(self):
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self):
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config tha are passed to the flax_model when it's build in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')
