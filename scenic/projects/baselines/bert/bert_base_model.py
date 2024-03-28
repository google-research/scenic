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

"""Base class for models working with bert."""

from typing import Callable, Dict, Optional, Tuple, Union

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[Dict[str, jnp.ndarray], Batch], Dict[str, Tuple[float,
                                                                     int]]]
LossFn = Callable[[Dict[str, jnp.ndarray], Batch, Optional[jnp.ndarray]], float]


def num_examples(
    logits: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, int]:
  if weights is None:
    return logits.shape[0]
  return weights.sum()


def sparse_weighted_unnormalized_softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mlm_weights: jnp.ndarray,
    batch_mask_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes sparse weighted softmax cross entropy give logits and targets.

  Args:
    logits: Logits of shape [batch_size, length, vocab_size].
    labels: Labels from {0 ... vocab_size - 1} of shape [batch_size, length].
    mlm_weights: Weights of shape [batch_size, length], indicating masked tokens
      in masked language modeling task.
    batch_mask_weights: None or array of shape [batch,] indicating masked
      examples.

  Returns:
    Per example Loss value.
  """
  batch_size, length, vocab_size = logits.shape
  logits = jax.nn.log_softmax(logits)
  logits, mlm_weights = logits.ravel(), mlm_weights.ravel()
  offsets = (np.arange(batch_size * length) * vocab_size).reshape((-1, length))
  labels = (labels + offsets).ravel()
  loss = -jnp.take(logits, labels, axis=0)
  loss = loss * mlm_weights
  loss = loss.sum(axis=-1, keepdims=True) / (
      mlm_weights.sum(axis=-1, keepdims=True) + 1e-8
  )
  if batch_mask_weights is not None:
    loss = model_utils.apply_weights(loss, batch_mask_weights)

  return loss


def sparse_weighted_softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mlm_weights: jnp.ndarray,
    batch_mask_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Same as weighted_unnormalized, but additionally takes a mean.

  Args:
    logits: Logits of shape [batch_size, length, vocab_size].
    labels: Labels from {0 ... vocab_size - 1} of shape [batch_size, length].
    mlm_weights: Weights of shape [batch_size, length], indicating masked tokens
      in masked language modeling task.
    batch_mask_weights: None or array of shape [batch,] indicating masked
      examples.

  Returns:
    The mean cross entropy of the examples in the given batch as a scalar.
  """
  if batch_mask_weights is not None:
    normalization = batch_mask_weights.sum()
  else:
    normalization = mlm_weights.shape[0]  # Batch size.
  sparse_unnormalized_softmax_ce = (
      sparse_weighted_unnormalized_softmax_cross_entropy(  # pylint: disable=line-too-long
          logits, labels, mlm_weights, batch_mask_weights
      )
  )
  return jnp.sum(sparse_unnormalized_softmax_ce) / (normalization + 1e-8)


def sparse_weighted_per_example_accuracy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mlm_weights: jnp.ndarray,
    batch_mask_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes weighted number of correctly classified over the given batch.

  This computes the weighted number of correctly classified masked tokens in a
  single, potentially padded minibatch. If the minibatch/inputs is padded (i.e.,
  it contains null examples/pad pixels) it is assumed that batch_mask_weights
  is a binary mask where 0 indicates that the example/pixel is null/padded.
  We assume the trainer will aggregate and divide by number of samples.

  Args:
    logits: Logits of shape [batch_size, length, vocab_size].
    labels: Labels from {0 ... vocab_size - 1} of shape [batch_size, length].
    mlm_weights: Weights of shape [batch_size, length], indicating masked tokens
      in masked language modeling task.
    batch_mask_weights: None or array of shape [batch,] indicating masked
      examples.

  Returns:
    Per example accuracy of predicted masked tokens.
  """
  preds = jnp.argmax(logits, axis=-1)
  correct = jnp.equal(preds, labels)
  correct = correct * mlm_weights
  # Shape of per example acccuracy will be (batch_size,).
  per_ex_accuracy = correct.sum(axis=-1) / (mlm_weights.sum(axis=-1) + 1e-8)
  if batch_mask_weights is not None:
    per_ex_accuracy = model_utils.apply_weights(per_ex_accuracy,
                                                batch_mask_weights)
  return per_ex_accuracy


def bert_metrics_function(outputs: Dict[str, jnp.ndarray],
                          batch: Batch) -> Dict[str, Tuple[float, int]]:
  """Calcualte metrics for the BERT task.

  Args:
   outputs: Output of model that has masked LM logits of shape [batch, length,
     vocab_size], and  next sentence prediction logits of shape [batch, 2].
   batch: Batch of data that has 'masked_lm_ids', 'masked_lm_weights' and
     'next_sentence_labels'.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  mlm_logits = outputs['mlm_logits']
  nsp_logits = outputs['nsp_logits']
  next_sentence_labels = common_utils.onehot(batch['next_sentence_labels'], 2)
  batch_weights = batch.get('batch_mask')  # batch_mask might not be defined
  per_ex_nsp_loss = model_utils.weighted_unnormalized_softmax_cross_entropy(
      nsp_logits, next_sentence_labels, batch_weights)
  per_ex_nsp_accuracy = model_utils.weighted_correctly_classified(
      nsp_logits, next_sentence_labels, batch_weights)

  per_ex_mlm_loss = sparse_weighted_unnormalized_softmax_cross_entropy(
      mlm_logits, batch['masked_lm_ids'], batch['masked_lm_weights'],
      batch_weights)
  per_ex_mlm_accuracy = sparse_weighted_per_example_accuracy(
      mlm_logits, batch['masked_lm_ids'], batch['masked_lm_weights'],
      batch_weights)

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  normalizer = num_examples(mlm_logits, batch_weights)
  for name, value in zip(
      ['nsp_loss', 'nsp_accuracy', 'mlm_loss', 'mlm_accuracy', 'loss'], [
          per_ex_nsp_loss, per_ex_nsp_accuracy, per_ex_mlm_loss,
          per_ex_mlm_accuracy, per_ex_nsp_loss + per_ex_mlm_loss
      ]):
    evaluated_metrics[name] = model_utils.psum_metric_normalizer(
        (value, normalizer))
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


def compute_bert_loss(mlm_logits: jnp.ndarray, nsp_logits: jnp.ndarray,
                      batch: Batch) -> float:
  """Computes BERT loss.

  Args:
   mlm_logits: Masked LM logits of shape [batch, length, vocab_size].
   nsp_logits: Next sentence prediction logits of shape [batch, 2].
   batch: Batch of data that has 'masked_lm_ids', 'masked_lm_weights' and
     'next_sentence_labels'.

  Returns:
    Loss value.
  """
  next_sentence_labels = common_utils.onehot(batch['next_sentence_labels'], 2)
  batch_weights = batch.get('batch_mask')  # batch_mask might not be defined
  nsp_loss = model_utils.weighted_softmax_cross_entropy(nsp_logits,
                                                        next_sentence_labels,
                                                        batch_weights)
  mlm_loss = sparse_weighted_softmax_cross_entropy(mlm_logits,
                                                   batch['masked_lm_ids'],
                                                   batch['masked_lm_weights'],
                                                   batch_weights)
  return nsp_loss + mlm_loss  # pytype: disable=bad-return-type  # jax-ndarray


class BERTBaseModel(base_model.BaseModel):
  """Defines BERT base models.

  A model is class with three members: get_metrics_fn, loss_fn, and a
  flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```metric_fn(logits, label, weights).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).

  This model class defines a softmax_cross_entropy_loss with weight decay,
  where the weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = bert_model.BERTModel
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model
    dummy_input = {name: jnp.zeros(input_shape, model_input_dtype), ...}
    model_state, params = flax_model.init(
        rng, dummy_input, train=False).pop('params')
    ```
  And this is how to call the model:s
    ```
    variables = {'params': params, **model_state}
    output, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def get_metrics_fn(self, split: Optional[str] = None) -> MetricFn:  # pytype: disable=signature-mismatch  # jax-ndarray
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(outputs,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return bert_metrics_function

  def loss_function(self,
                    outputs: Dict[str, jnp.ndarray],
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      outputs: a dictionary containing either 'logits' key of shape [batch,
        length, num_classes] or 'nsp_logits' of shape [batch, 2] and
        'mlm_logits' of shape [batch, length, vocab_size] (for 'BERT' task).
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    total_loss = compute_bert_loss(outputs['mlm_logits'], outputs['nsp_logits'],
                                   batch)

    if self.config.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss += 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss

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
