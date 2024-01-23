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

"""Utils for linear probe evaluation."""

import functools
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Text, Tuple, Type, Union

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils as scenic_model_utils
from scenic.train_lib import classification_trainer
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils


PyTree = Union[Mapping[str, Mapping], Any]
Batch = Dict[str, jnp.ndarray]
Metric = Dict[str, Tuple[float, int]]


class LinearProbe(nn.Module):
  """A linear probe."""
  num_classes: int

  @nn.compact
  def __call__(self, x, train: bool = False, debug: bool = False):
    del train, debug
    x = jax.lax.stop_gradient(x)
    x = nn.Dense(features=self.num_classes)(x)
    return x


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    representation_fn: Callable[[Batch], jnp.ndarray],
    linear_probe: nn.Module,
    label_smoothing: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]]]:
  """The training step.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer.
    batch: A single batch of data. Must contain the following keys:
      'representations': A [batch_size, representation_size] ndarray of the
      outputs of the base network, which will be fed into the linear probe.
      'label': A [batch_size] ndarray of integer class labels. 'batch_mask': A
      [batch_size] ndarray where 1.0 indicates a valid sample and 0.0 indicates
      an invalid sample.
    representation_fn: A function that given a batch, returns representations
      that are passed to the linear_probe_fn as input.
    linear_probe: The linear probe module.
    label_smoothing: The label smoothing coefficient.
    max_grad_norm: Maximum gradient norm used for gradient clliping.

  Returns:
    The updated train state.
  """

  def loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = linear_probe.apply(
        variables, representation_fn(batch), mutable=['batch_stats'], train=True
    )
    loss = scenic_model_utils.weighted_softmax_cross_entropy(
        logits,
        batch['label'],
        batch['batch_mask'],
        label_smoothing=label_smoothing)
    return loss, (new_model_state, logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (train_cost, (new_model_state, logits)), grad = grad_fn(train_state.params)
  del train_cost

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if max_grad_norm is not None:
    grad = clip_grads(grad, max_grad_norm)

  tx = train_state.tx
  if tx is None:
    raise ValueError('train_state.tx, the Gradient Transformation, is None')

  updates, new_opt_state = tx.update(
      grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
  )
  metrics = classification_model.classification_metrics_function(
      logits, batch, target_is_onehot=True
  )
  return new_train_state, metrics


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    representation_fn: Callable[[Batch], jnp.ndarray],
    linear_probe: nn.Module,
    metrics_fn: classification_trainer.MetricFn,
) -> Metric:
  """Runs a single step of training.

  Assumed API of metrics_fn is:
  ```metrics = metrics_fn(logits, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. eval_step will
  aggregate (by summing) all per example measurements and divide by the
  aggregated normalizers. For each given metric we compute:
  1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
  over all batches.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data.
    representation_fn: A function that given a batch, returns representations
      that are passed to the linear_probe_fn as input.
    linear_probe: The linear probe module.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.

  Returns:
    Calculated metrics.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  logits = linear_probe.apply(variables, representation_fn(batch))
  metrics = metrics_fn(logits, batch)
  return metrics


class LinearEvaluator:
  """Class for linear evaluation.

  Attributes:
    representation_fn: A function that is passed a TrainState and a Batch, and
      returns a tuple of (representation, label, and mask) for that batch. label
      and mask are ignored.
    rng: A PRNG that will be used to preprocess the data and to initialize the
      linear probe.
    config: A ConfigDict.
    linear_probe_cls: A Flax Module that is used as the linear probe. It must
      have a constructor argument called `num_classes`. Other constructor
      arguments are passed through via `linear_probe_init_kwargs`.
    linear_probe_init_kwargs: Keyword arguments passed to the constructor of
      `linear_probe_cls`.
  """

  def __init__(self,
               representation_fn: Callable[[train_utils.TrainState, Batch],
                                           Tuple[jnp.ndarray, Any, Any]],
               rng: jnp.ndarray,
               linear_eval_config: ml_collections.ConfigDict,
               linear_probe_cls: Type[nn.Module] = LinearProbe,
               linear_probe_init_kwargs: Optional[Dict[str, Any]] = None):
    self.representation_fn = representation_fn
    self.rng = rng
    self.config = linear_eval_config  # Shared configs among all datasets.
    self.linear_probe_cls = linear_probe_cls
    self.linear_probe_init_kwargs = linear_probe_init_kwargs or {}
    self._datasets = {}  # This will be our cache for lazy loading.

  def _get_dataset(self, ds_name: str, config: ml_collections.ConfigDict,
                   rng: jnp.ndarray) -> dataset_utils.Dataset:
    """Lazy-loads given dataset."""
    try:
      return self._datasets[ds_name]
    except KeyError:
      dataset = train_utils.get_dataset(config, rng)
      return self._datasets.setdefault(ds_name, dataset)

  def _train(self,
             representation_fn: Callable[[Batch], jnp.ndarray],
             linear_probe: nn.Module,
             dataset: dataset_utils.Dataset,
             rng: jnp.ndarray,
             config: ml_collections.ConfigDict,
             ds_name: str,
             writer: Optional[metric_writers.MetricWriter] = None,
             repr_step: int = 0) -> train_utils.TrainState:
    """The main training loop.

    Args:
      representation_fn: A function that given a batch, returns representations
        that are passed to the linear_probe_fn as input.
      linear_probe: The linear probe module instance.
      dataset: The dataset that has train_iter, valid_iter, meta_data, and
        optionally, test_iter.
      rng: JAX rng key.
      config: Configurations of the optimizer and learning rate scheduler.
      ds_name: The name of the training dataset.
      writer: A metric writer. Only needed if train summaries are to be written.
      repr_step: The training step of the representation model being evaluated.

    Returns:
      The train state of the trained linear probe.
    """
    # Initialize model.
    input_shape = [1] + list(dataset.meta_data['input_shape'][1:])
    dummy_reprs = representation_fn({
        'inputs': jnp.zeros(input_shape),
        'label': None,
        'batch_mask': None
    })
    model_state, params = flax.core.pop(
        linear_probe.init({'params': rng}, dummy_reprs), 'params'
    )

    # Create optimizer.
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    optimizer_config = optimizers.get_optax_optimizer_config(config)
    # If the config is already an optax-compatible config, better call directly:
    #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)
    tx = optimizers.get_optimizer(optimizer_config, lr_fn)
    # We jit this, such that the arrays that are created are created on the same
    # device as the input is, in this case the CPU. Else they'd be on device[0].
    opt_state = jax.jit(tx.init, backend='cpu')(params)

    train_state = train_utils.TrainState(
        global_step=0,
        opt_state=opt_state,
        tx=tx,
        params=params,
        model_state=model_state,
    )
    train_state = jax_utils.replicate(train_state)
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            representation_fn=representation_fn,
            linear_probe=linear_probe,
            label_smoothing=config.get('label_smoothing'),
            max_grad_norm=config.get('max_grad_norm'),
        ),
        donate_argnums=(0, 1),
        axis_name='batch',
    )
    # Calculate the total number of training steps.
    total_steps, _ = train_utils.get_num_training_steps(config,
                                                        dataset.meta_data)
    train_metrics = []
    extra_train_logs = []
    prefix = f'linear_eval_train/{ds_name}/step_{repr_step}'
    for step in range(total_steps):
      batch = next(dataset.train_iter)
      train_state, metrics = p_train_step(train_state, batch)
      train_metrics.append(metrics)
      # TODO(scenic-dev): Figure out how to get the lr from the optimizer.
      extra_train_logs.append({f'{prefix}/learning_rate': lr_fn(step)})
      if (step % 1000 == 0) or (step == total_steps - 1):
        logging.info('Linear probe trained for %d steps.', step)
      if (writer and self.config.get('log_train_summary_steps', 0) > 0 and
          step % self.config.log_train_summary_steps == 0):
        train_utils.log_train_summary(
            step=step,
            train_metrics=jax.tree_util.tree_map(
                train_utils.unreplicate_and_get, train_metrics),
            extra_training_logs=jax.tree_util.tree_map(jax.device_get,
                                                       extra_train_logs),
            writer=writer,
            prefix=prefix,
            key_separator='/')
        train_metrics = []
        extra_train_logs = []
    logging.info('Linear probe training complete for dataset %s.', ds_name)
    return train_state

  def _eval(self, representation_fn: Callable[[Batch], jnp.ndarray],
            linear_probe: nn.Module, dataset: dataset_utils.Dataset,
            train_state: train_utils.TrainState,
            config: ml_collections.ConfigDict, ds_name: str,
            writer: metric_writers.MetricWriter,
            repr_step: int) -> List[Metric]:
    """Evaluates a trained linear probe.

    Args:
      representation_fn: A function that given a batch, returns representations
        that are passed to the linear_probe_fn as input.
      linear_probe: The linear probe module instance.
      dataset: The dataset that has train_iter, valid_iter, meta_data, and
        optionally, test_iter.
      train_state: The train state of the trained LinearProbe.
      config: Configurations of evaluation, e.g., batch size.
      ds_name: The name of the training dataset.
      writer: A metric writer.
      repr_step: The training step of the representation model being evaluated.

    Returns:
      A list of metrics computed over each batch of test data.
    """
    p_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            representation_fn=representation_fn,
            linear_probe=linear_probe,
            metrics_fn=functools.partial(
                classification_model.classification_metrics_function,
                target_is_onehot=dataset.meta_data['target_is_onehot'])),
        donate_argnums=(1,),
        axis_name='batch')
    eval_metrics = []
    eval_batch_size = config.get('eval_batch_size', config.batch_size)
    total_eval_steps = int(
        np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
    for _ in range(total_eval_steps):
      batch = next(dataset.valid_iter)
      metrics = p_eval_step(train_state, batch)
      eval_metrics.append(train_utils.unreplicate_and_get(metrics))

    train_utils.log_eval_summary(
        step=repr_step,
        eval_metrics=eval_metrics,
        writer=writer,
        prefix=f'linear_eval/{ds_name}')

    return eval_metrics

  def run_one_dataset(self, ds_name: str, config: ml_collections.ConfigDict,
                      representation_fn: Callable[[Batch], jnp.ndarray],
                      rng: jnp.ndarray, writer: metric_writers.MetricWriter,
                      repr_step: int) -> List[Metric]:
    """Computes linear evaluation metrics on one dataset.

    Args:
      ds_name: Name of the dataset, used for loading the data from cache dict.
      config: Configuration of the dataset, train and evaluation on.
      representation_fn: A function that given a batch, returns representations
        that are passed to the linear_probe_fn as input.
      rng: The JAX rng key.
      writer: A metric writer. Only needed if train summaries are to be written.
      repr_step: The training step of the representation model being evaluated.

    Returns:
      A list of metrics computed over each batch of test data.
    """
    data_rng, train_rng = jax.random.split(rng)
    dataset = self._get_dataset(ds_name, config, data_rng)
    linear_probe_head = self.linear_probe_cls(
        num_classes=dataset.meta_data['num_classes'],
        **self.linear_probe_init_kwargs)
    logging.info('[linear_eval]: Training linear probe for dataset %s', ds_name)
    train_state = self._train(representation_fn, linear_probe_head, dataset,
                              train_rng, config, ds_name, writer, repr_step)
    logging.info('[linear_eval]: Evaluating linear probe for dataset %s',
                 ds_name)
    eval_metrics = self._eval(representation_fn, linear_probe_head, dataset,
                              train_state, config, ds_name, writer, repr_step)
    return eval_metrics

  def run_all(self, repr_train_state: Any,
              datasets: Dict[Text, ml_collections.ConfigDict],
              writer: metric_writers.MetricWriter,
              repr_step: int) -> Dict[Text, List[Metric]]:
    """Computes linear evaluation metrics over multiple datasets.

    Args:
      repr_train_state: The train state that should be passed in as the first
        argument to `representation_fn`.
      datasets: A dictionary of datasets to evaluate on. The keys are names of
        the dataset that are used as the keys of the output dictionary. The
        values are configurations for the linear probe.
      writer: A metric writer.
      repr_step: The training step of the representation model being evaluated.

    Returns:
      A dictionary where the keys are the same as `datasets` and the values are
      lists of metrics computed over each test batch of that dataset.
    """

    # Prepare the representation function given the current state of training.
    def representation_fn(batch: Batch) -> jnp.ndarray:
      return self.representation_fn(
          train_state=jax_utils.unreplicate(repr_train_state), batch=batch)[0]  # pytype: disable=wrong-keyword-args

    results = {}
    for ds_name, ds_cfg in datasets.items():
      logging.info('[linear_eval][%s]', ds_name)
      self.rng, ds_rng = jax.random.split(self.rng)
      results[ds_name] = self.run_one_dataset(ds_name, ds_cfg,
                                              representation_fn, ds_rng, writer,
                                              repr_step)

    return results
