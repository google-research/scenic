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

"""SVViT Inference Script."""

import datetime
import functools
import os
import pickle
from typing import Any, Callable, Optional, Type

from absl import logging
from clu import metric_writers
from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.svvit import classification_trainer as trainer
from scenic.projects.svvit import metrics as sv_metric
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
import tensorflow as tf


# Aliases for custom types:
Batch = dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, dict[str, jnp.ndarray]],
                    dict[str, tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def restore_train_state(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model: Any,
    dataset: dataset_utils.Dataset,
):
  """Initializes the model state."""
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, _, _) = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(dataset.meta_data['input_shape'],
                   dataset.meta_data.get('input_dtype', jnp.float32))],
      config=config,
      rngs=init_rng)
  # Create optimizer.
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  # If the config is already an optax-compatible config, better call directly:
  #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)
  tx = optimizers.get_optimizer(optimizer_config, 0, params=params)
  # We jit this, such that the arrays that are created on the same device as the
  # input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={})
  init_checkpoint_path = config.init_from.get('checkpoint_path')
  restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, train_state, assert_exist=True)
  current_step = restored_train_state.global_step
  logging.info(
      'Parameter summary after initialising from restored train state '
      'at step %d:', current_step)
  debug_utils.log_param_shapes(restored_train_state.params)
  return restored_train_state, current_step


def inference_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    debug: Optional[bool] = False
):
  """Runs a single step of training."""
  variables = {'params': train_state.params, **train_state.model_state}
  capture_intermediates = lambda mdl, _: mdl.name == 'pre_logits'
  logits, intermediate = flax_model.apply(
      variables,
      batch['inputs'],
      train=False,
      mutable=False,
      debug=debug,
      capture_intermediates=capture_intermediates,
  )
  return nn.softmax(
      logits,
      axis=-1), intermediate['intermediates']['pre_logits']['__call__'][0]


def compute_similarity_scores(
    train_state: train_utils.TrainState,
    iterator,
    eval_step_fn,
    eval_steps,
    workdir,
    lead_host,
):
  """Computes similarity scores and dump them directly instead of metrics."""
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)
  all_logits, all_keys, all_labels, all_batch_masks, all_intermediate = [], [], [], [], []
  # Do this to ensure we definitely cover the full test set
  eval_steps = int(np.ceil(1.3 * eval_steps))
  logging.info('Number of eval steps is %s', eval_steps)
  for step in range(eval_steps):
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      eval_batch = next(iterator)
      assert 'key' in eval_batch, 'Keys must be added to batch'
      keys = eval_batch['key']
      labels = eval_batch['label']
      batch_masks = eval_batch['batch_mask']
      del eval_batch['key']
      del eval_batch['label']

      logits, intermediate = eval_step_fn(train_state, eval_batch)
      gathered_logits, gathered_keys, gathered_labels, gathered_batch_masks, gathered_intermediate = all_gather_and_unreplicate(
          (logits, keys, labels, batch_masks, intermediate))
      all_logits.append(np.concatenate(gathered_logits, axis=0))
      all_intermediate.append(np.concatenate(gathered_intermediate, axis=0))
      all_labels.append(np.concatenate(gathered_labels, axis=0))
      all_keys.append(
          tf.strings.unicode_encode(
              np.concatenate(gathered_keys, axis=0), 'UTF-8'))
      all_batch_masks.append(np.concatenate(gathered_batch_masks, axis=0))

  logging.info('all_scores.shape: %s', str(len(all_keys)))

  timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  fname_logits = os.path.join(workdir, f'logits_offline_eval_{timestamp}')
  fname_intermediate = os.path.join(workdir,
                                    f'intermediate_offline_eval_{timestamp}')
  fname_labels = os.path.join(workdir, f'labels_offline_eval_{timestamp}')
  fname_keys = os.path.join(workdir, f'keys_offline_eval_{timestamp}')
  fname_masks = os.path.join(workdir, f'masks_offline_eval_{timestamp}')
  if lead_host:
    logging.info('Logging results to %s', fname_logits)
    log_to_cns(
        predictions=np.concatenate(all_logits, axis=0),
        filename_prefix=fname_logits)
    log_to_cns(
        predictions=np.concatenate(all_intermediate, axis=0),
        filename_prefix=fname_intermediate)
    log_to_cns(
        predictions=np.concatenate(all_keys, axis=0),
        filename_prefix=fname_keys)
    log_to_cns(
        predictions=np.concatenate(all_labels, axis=0),
        filename_prefix=fname_labels)
    log_to_cns(
        predictions=np.concatenate(all_batch_masks, axis=0),
        filename_prefix=fname_masks)


def log_to_cns(predictions, filename_prefix: str):
  """Saves predictions to CNS.

  Args:
    predictions: Serialised predictions.
    filename_prefix: File prefix to save the results to.
  """
  with open(filename_prefix + '.pkl', 'wb') as f:
    # Protocol needs to be set to save large files.
    pickle.dump(predictions, f, protocol=4)


def all_gather_and_unreplicate(inputs):
  return jax_utils.unreplicate(
      jax.pmap(lambda x: jax.lax.all_gather(x, 'batch'), 'batch')(inputs))


def evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> dict[str, Any]:
  """Evaluates the model.

  This function loads a pretrained model, optionally overrides some arguments
  related to evaluation in its original config, and then evaluates the model
  on the specified dataset.

  Args:
    rng: Jax rng key.
    config: Configurations for evaluation. Can be reused to override some
      settings from the training config.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
     eval_summary: Dictionary with the evaluation summary
  """
  lead_host = jax.process_index() == 0

  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)
  # Initialize model.
  train_state, current_step = restore_train_state(rng, config, model, dataset)
  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  eval_step_pmapped = jax.pmap(
      functools.partial(
          trainer.eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          all_gather=config.get('global_metrics', False),
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  inference_step_pmapped = jax.pmap(
      functools.partial(
          inference_step,
          flax_model=model.flax_model,
          debug=config.debug_eval),
      axis_name='batch',
      donate_argnums=(1,),
  )
  # If `global_metrics` are set in the config and we are the lead host
  compute_global_metrics = False
  if config.get('global_metrics', False) and lead_host:
    compute_global_metrics = True
  if compute_global_metrics:
    global_metrics_evaluator = sv_metric.TruvariGlobalEvaluator(
        config.global_metrics)
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)
  eval_summary = {}
  # Ceil rounding such that we include the last incomplete batch.
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size))
  eval_metrics = []
  if not config.save_predictions_on_cns:
    for s in range(total_eval_steps):
      eval_batch = next(dataset.valid_iter)
      e_metrics, e_output, e_batch = eval_step_pmapped(train_state, eval_batch)
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      logging.info('eval metircs at step %d', s)
      if compute_global_metrics:
        # Unreplicate outputs of eval_step_pmapped that are coming from
        # `lax.all_gather`, fetch to the host and add to the Evaluator:
        e_batch_mask = train_utils.unreplicate_and_get(
            e_batch['batch_mask']).astype(bool)
        global_metrics_evaluator.add_batch_of_examples(
            target=train_utils.unreplicate_and_get(
                e_batch['label'])[e_batch_mask],
            output=train_utils.unreplicate_and_get(e_output)[e_batch_mask])
        del e_batch, e_output, e_batch_mask
    eval_global_metrics_summary = None
    if compute_global_metrics:
      if dataset.meta_data['num_eval_examples'] != len(
          global_metrics_evaluator):
        logging.warning(
            'Number of eval (valid/test) examples in the dataset metadata is '
            '%d, however the global evaluator captured only %d of them',
            dataset.meta_data['num_eval_examples'],
            len(global_metrics_evaluator))
      eval_global_metrics_summary = (
          global_metrics_evaluator.compute_metrics(clear_annotations=True))

    eval_summary.update(
        train_utils.log_eval_summary(
            step=current_step,
            eval_metrics=eval_metrics,
            extra_eval_summary=eval_global_metrics_summary,
            writer=writer,
            prefix='SV_test'))
    del eval_metrics, eval_global_metrics_summary
  else:
    compute_similarity_scores(
        train_state=train_state,
        iterator=dataset.valid_iter,
        eval_step_fn=inference_step_pmapped,
        eval_steps=total_eval_steps,
        workdir=workdir,
        lead_host=lead_host)
  writer.flush()
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return eval_summary
