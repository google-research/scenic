"""SVViT Inference Script."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from flax import jax_utils
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
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def restore_train_state(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model: Any,
    dataset: dataset_utils.Dataset,
):
  """Initialize the model state."""
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, _, _) = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(dataset.meta_data['input_shape'],
                   dataset.meta_data.get('input_dtype', jnp.float32))],
      config=config,
      rngs=init_rng)
  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  del params  # Do not keep a copy of the initial params.
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
  init_checkpoint_path = config.init_from.get('checkpoint_path')
  restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, train_state, assert_exist=True)
  current_step = restored_train_state.global_step
  logging.info(
      'Parameter summary after initialising from restored train state '
      'at step %d:', current_step)
  debug_utils.log_param_shapes(restored_train_state.optimizer.target)
  return restored_train_state, current_step


def evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
) -> Dict[str, Any]:
  """Evaluate the model.

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
  for s in range(total_eval_steps):
    eval_batch = next(dataset.valid_iter)
    e_metrics, e_output, e_batch = eval_step_pmapped(train_state, eval_batch)
    eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
    logging.info('eval metircs at step %d', s)
    logging.info(eval_metrics)
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
    if dataset.meta_data['num_eval_examples'] != len(global_metrics_evaluator):
      logging.warning(
          'Number of eval (valid/test) examples in the dataset metadata is '
          '%d, however the global evaluator captured only %d of them',
          dataset.meta_data['num_eval_examples'], len(global_metrics_evaluator))
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
  writer.flush()
  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return eval_summary
