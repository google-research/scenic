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

"""Eval functionality for DeformableDETR."""

from concurrent import futures
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from absl import logging
from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib.dataset_utils import Dataset
from scenic.projects.baselines.deformable_detr.coco_eval import DeformableDetrGlobalEvaluator
from scenic.projects.baselines.deformable_detr.coco_eval import prepare_coco_eval_dicts
from scenic.projects.baselines.detr import train_utils as detr_train_utils
from scenic.train_lib import train_utils

ArrayDict = Dict[str, jnp.ndarray]


def get_eval_step(
    flax_model: nn.Module,
    loss_and_metrics_fn: Callable[..., Any],
    logits_to_probs_fn: Callable[[jnp.ndarray], jnp.ndarray],
    metrics_only: bool = False,
    debug: bool = False
) -> Callable[[train_utils.TrainState, ArrayDict], Tuple[Any, Any, Any]]:
  """Runs a single step of training.


  Args:
    flax_model: Instance of model to evaluate.
    loss_and_metrics_fn: A function that given model predictions, a batch, and
      parameters of the model calculates the loss as well as metrics.
    logits_to_probs_fn: Function that takes logits and converts them to probs.
    metrics_only: Only return metrics.
    debug: bool; Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Eval step function which returns predictions and calculated metrics. Also
    the buffer of the second argument (batch) is donated to the computation.
  """

  def metrics_fn(train_state: train_utils.TrainState, batch: ArrayDict,
                 predictions: ArrayDict) -> Tuple[Any, Any, Any]:
    _, metrics = loss_and_metrics_fn(
        predictions, batch, model_params=train_state.params)

    if metrics_only:
      return None, None, metrics

    targets, predictions_out = prepare_coco_eval_dicts(
        batch=batch,
        predictions=predictions,
        logits_to_probs_fn=logits_to_probs_fn,
        gather=True)
    return targets, predictions_out, metrics

  def eval_step(train_state: train_utils.TrainState,
                batch: ArrayDict) -> Tuple[Any, Any, Any]:
    variables = {
        'params': train_state.params,
        **train_state.model_state
    }
    predictions = flax_model.apply(
        variables,
        batch['inputs'],
        padding_mask=batch['padding_mask'],
        train=False,
        mutable=False,
        debug=debug)
    return metrics_fn(train_state, batch, predictions)

  return eval_step


def _wait(future: Optional[futures.Future]) -> Any:  # pylint: disable=g-bare-generic
  if future is None:
    return None
  return future.result()


def _add_examples(global_metrics_evaluator: DeformableDetrGlobalEvaluator,
                  predictions: Sequence[ArrayDict],
                  labels: Sequence[ArrayDict]):
  for pred, label in zip(predictions, labels):
    global_metrics_evaluator.add_example(prediction=pred, target=label)  # pytype: disable=wrong-arg-types  # jax-ndarray


def run_eval(
    global_metrics_evaluator: DeformableDetrGlobalEvaluator, dataset: Dataset,
    train_state: train_utils.TrainState,
    eval_step_pmapped: Callable[[train_utils.TrainState, ArrayDict], Any],
    pool: futures.ThreadPoolExecutor, step: int, steps_per_eval: int
) -> Tuple[Tuple[int, Any], futures.Future]:  # pylint: disable=g-bare-generic
  """Run full eval on dataset."""
  future = None

  eval_metrics = []
  if global_metrics_evaluator is not None:
    global_metrics_evaluator.clear()

  for eval_step in range(steps_per_eval):
    logging.info('Running eval step %d', eval_step)
    eval_batch = next(dataset.valid_iter)

    # Do the eval step given the matches.
    (eval_batch_all_hosts, eval_predictions_all_hosts,
     e_metrics) = eval_step_pmapped(train_state, eval_batch)

    # Variable aux_outputs is not needed anymore.
    eval_predictions_all_hosts.pop('aux_outputs', None)

    # Collect local metrics (returned by the loss function).
    eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

    if global_metrics_evaluator is not None:
      # Unreplicate the output of eval_step_pmapped (used `lax.all_gather`).
      eval_batch_all_hosts = jax_utils.unreplicate(eval_batch_all_hosts)
      eval_predictions_all_hosts = jax_utils.unreplicate(
          eval_predictions_all_hosts)

      # Collect preds and labels to be sent for computing global metrics.
      predictions = detr_train_utils.process_and_fetch_to_host(
          eval_predictions_all_hosts, eval_batch_all_hosts['batch_mask'])
      predictions = jax.tree_util.tree_map(np.asarray, predictions)

      labels = detr_train_utils.process_and_fetch_to_host(
          eval_batch_all_hosts['label'], eval_batch_all_hosts['batch_mask'])
      labels = jax.tree_util.tree_map(np.asarray, labels)

      if eval_step == 0:
        logging.info('Pred keys: %s', list(predictions[0].keys()))
        logging.info('Labels keys: %s', list(labels[0].keys()))

      # Add to evaluator.
      _wait(future)
      future = pool.submit(_add_examples, global_metrics_evaluator, predictions,
                           labels)

      del predictions, labels

    del eval_batch, eval_batch_all_hosts, eval_predictions_all_hosts

  eval_global_metrics_summary_future = None
  if global_metrics_evaluator is not None:
    _wait(future)
    logging.info('Number of eval examples: %d', len(global_metrics_evaluator))
    eval_global_metrics_summary_future = pool.submit(
        global_metrics_evaluator.compute_metrics, clear_annotations=False)

  return (step, eval_metrics), eval_global_metrics_summary_future
