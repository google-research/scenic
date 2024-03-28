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

"""Evaluation script for the CenterNet.

This file is modified from scenic DETR code at
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/
detr/trainer.py
"""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.baselines.centernet import evaluators
from scenic.projects.baselines.centernet import train_utils as centernet_train_utils
from scenic.train_lib import train_utils


def eval_step(
    train_state, batch, *,
    flax_model, postprocess=False, debug=False):
  """Runs a single step of inference."""
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }
  predictions = flax_model.apply(
      variables,
      batch['inputs'],
      preprocess=True,
      postprocess=postprocess,
      padding_mask=batch['padding_mask'],
      train=False,
      mutable=False,
      debug=debug)
  # metrics (losses, etc.) are disabled for testing, as the model directly
  #   return post-processed outputs.
  metrics = {}
  targets = {'label': batch['label'], 'batch_mask': batch['batch_mask']}
  predictions = jax.lax.all_gather(predictions, 'batch')
  targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions, metrics


def inference_on_dataset(
    flax_model: Any,
    train_state: train_utils.TrainState,
    dataset: dataset_utils.Dataset,
    eval_batch_size: int = 1,
    is_host: bool = False,
    save_dir: str = '',
    config: ml_collections.ConfigDict = ml_collections.ConfigDict(),
    ) -> Any:
  """The main evaluation loop. Run evaluation on the whole validation set.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    train_state: train_state that contains the model parameters.
    dataset: The dataset that has valid_iter and meta_data.
    eval_batch_size: integer. Batch size per-device in evaluation.
    is_host: bool: whether its the host machine. During multi-machine training,
      we only hold the evaluating data in one of the machines. The machine with
      `jax.process_index() == 0` sets `is_host` to True and will gather data
      from other machines and do the evaluation. Other machines set `is_host`
      as False.
    save_dir: string: where to save the json prediction.
    config: config dict.
  Returns:
    evaluation results.
  """
  annotations_loc = config.get('dataset_configs', {}).get(
      'test_annotation_path', None)
  eval_class_agnostic = config.get('eval_class_agnostic', False)
  eval_step_multiplier = config.get('eval_step_multiplier', 1.3)
  debug = config.get('debug_eval', False)
  global_metrics_evaluator = None  # Only run eval on the is_host node.
  if is_host:
    global_metrics_evaluator = evaluators.DetectionEvaluator(
        'lvis' if annotations_loc and ('lvis' in annotations_loc) else 'coco',
        annotations_loc=annotations_loc)
    global_metrics_evaluator.clear()

  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=flax_model,
          postprocess=True,
          debug=debug,
      ),
      axis_name='batch', donate_argnums=(1,),
  )

  eval_metrics = []
  total_eval_steps = int(np.ceil(eval_step_multiplier * dataset.meta_data[
      'num_eval_examples'] / eval_batch_size))
  for eval_step_i in range(total_eval_steps):
    if eval_step_i % 100 == 0:
      logging.info('Running eval step %d', eval_step_i)
    eval_batch = next(dataset.valid_iter)

    eval_batch_all_hosts, predictions_all_hosts, metrics = eval_step_pmapped(
        train_state, eval_batch)
    eval_metrics.append(train_utils.unreplicate_and_get(metrics))

    if global_metrics_evaluator is not None:
      eval_batch_all_hosts = jax_utils.unreplicate(eval_batch_all_hosts)
      predictions_all_hosts = jax_utils.unreplicate(
          predictions_all_hosts)

      # Collect preds and labels to be sent for computing global metrics.
      labels = centernet_train_utils.split_batch_and_fetch_to_host(
          eval_batch_all_hosts['label'], eval_batch_all_hosts['batch_mask'])
      labels = jax.tree_util.tree_map(np.asarray, labels)

      # concate per-device batch
      predictions_all_hosts = [
          jnp.concatenate(
              [predictions_all_hosts[b][i][:, None] for b in range(
                  len(predictions_all_hosts))], axis=1) for i in range(3)]
      results = centernet_train_utils.split_batch_and_fetch_to_host(
          predictions_all_hosts, eval_batch_all_hosts['batch_mask'])

      for pred, label in zip(results, labels):
        global_metrics_evaluator.add_example(prediction=pred, target=label)

  results = None
  if global_metrics_evaluator is not None:
    logging.info('Number of eval examples: %d', len(global_metrics_evaluator))
    if save_dir:
      global_metrics_evaluator.write_pred_annotations_to_file(
          save_dir, clear_annotations=False)
    results = global_metrics_evaluator.compute_metrics(
        clear_annotations=True, eval_class_agnostic=eval_class_agnostic)
  return results, eval_metrics


def evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
):
  """Prepares the items needed to run the evaluation.

  Args:
    rng: JAX PRNGKey.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.
  """
  is_host = jax.process_index() == 0
  model_config = config
  checkpoint_path = config.weights
  model = model_cls(model_config, dataset.meta_data)

  checkpoint_data = checkpoints.restore_checkpoint(checkpoint_path, None)
  params = checkpoint_data['params']
  model_state = {}
  if 'batch_stats' in checkpoint_data:  # For models converted from pytorch.
    model_state['batch_stats'] = checkpoint_data['batch_stats']
  elif 'model_state' in checkpoint_data:  # For saved scenic checkpoints.
    if 'batch_stats' in checkpoint_data['model_state']:
      model_state = flax.core.FrozenDict(
          {'batch_stats': checkpoint_data['model_state']['batch_stats']})
  train_state = train_utils.TrainState(
      global_step=0,
      params=flax.core.FrozenDict(params),
      model_state=flax.core.FrozenDict(model_state),
      rng=rng)
  train_state = jax_utils.replicate(train_state)
  del checkpoint_data, params, model_state

  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=0, writer=writer)

  hooks = []
  if is_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and is_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  start_time = time.time()
  with report_progress.timed('eval'):
    eval_results, eval_metrics = inference_on_dataset(
        model.flax_model,
        train_state,
        dataset,
        eval_batch_size=eval_batch_size,
        is_host=is_host,
        save_dir=workdir,
        config=config,
        )
    train_utils.log_eval_summary(
        step=0,
        eval_metrics=eval_metrics,
        extra_eval_summary=eval_results,
        writer=writer,
    )
  duration = time.time() - start_time
  logging.info('Done with evaluation: %.4f sec.', duration)
  writer.flush()
  train_utils.barrier()
