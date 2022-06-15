"""SVViT Inference Script."""

import functools
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
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

# Fields from the eval_config that we override in the original model config
_OVERRIDE_FIELDS = ('dataset_name', 'batch_size', 'dataset_configs',
                    'init_from')


def init_state(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
):
  """Initialize the model state."""
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  logging.info('The model has %d params, uses %d gflops', num_trainable_params,
               gflops)
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
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    if init_checkpoint_path is not None:
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
      # Load params from the init_model.
      train_state = model.init_from_train_state(  # pytype: disable=attribute-error
          train_state, restored_train_state, restored_model_cfg)
      del restored_train_state
  elif start_step == 0:
    logging.info('Training completely from scratch.'
                 'Not restoring from any checkpoint.')
  return train_state, start_step, num_trainable_params, gflops


def evaluate(
    *,
    rng: jnp.ndarray,
    eval_config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Dict[str, Any]:
  """Evaluate the model.

  This function loads a pretrained model, optionally overrides some arguments
  related to evaluation in its original config, and then evaluates the model
  on the specified dataset.

  Args:
    rng: Jax rng key.
    eval_config: Configurations for evaluation. Can be reused to override
      some settings from the training config.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    workdir: Directory to log metrics to.
    writer: CLU metrics writer instance.

  Returns:
     eval_summary: Dictionary with the evaluation summary
  """
  lead_host = jax.process_index() == 0

  config, checkpoint_dir = xm_utils.get_info_from_xmanager(
      eval_config.xid, eval_config.wid)
  if eval_config.get('workdir_as_checkpoint_dir'):
    workdir = checkpoint_dir

  # Override necessary configs.
  def maybe_overwrite_from_config(field):
    if field in eval_config:
      config[field] = eval_config[field]

  if eval_config.get('dataset_configs'):
    config.dataset_configs.update(eval_config.get('dataset_configs'))
  for field_name in _OVERRIDE_FIELDS:
    maybe_overwrite_from_config(field_name)
  logging.info('Evaluation config: %s', config)

  # Build dataset. This allows testing on a different dataset from training.
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(config, data_rng)

  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  train_state, _, _, _ = init_state(rng, config, model, dataset, workdir)

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
  assert config.dataset_configs.test_batch_size == jax.local_device_count(), (
      'The per-host batch size must be equal to the number of local devices.'
      'This ensures that each TPU device is processing different views of'
      'the same original video.')

  def evaluate_internal(
      train_state: train_utils.TrainState,
      step: int,
      valid_iter: Iterator[Batch],
      num_ex: int,
      val_name: str,
  ) -> Dict[str, Any]:
    eval_summary = {}
    # Ceil rounding such that we include the last incomplete batch.
    total_eval_steps = int(np.ceil(num_ex / config.batch_size))
    steps_per_eval = config.get('steps_per_eval') or total_eval_steps
    eval_metrics = []
    for _ in range(steps_per_eval):
      eval_batch = next(valid_iter)
      if dataset.meta_data['target_is_onehot']:  # Which includes multi-hot.
        # Ignore the entries with all zero label for evaluation.
        eval_batch['batch_mask'] *= eval_batch['label'].max(axis=-1)
      e_metrics, e_output, e_batch = eval_step_pmapped(
          train_state, eval_batch)
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      if compute_global_metrics:
        # Unreplicate outputs of eval_step_pmapped that are coming from
        # `lax.all_gather`, fetch to the host and add to the Evaluator:
        e_batch_mask = train_utils.unreplicate_and_get(
            e_batch['batch_mask']).astype(bool)
        # Classification: 'label', regression: 'target'
        t_key = 'label' if 'label' in e_batch else 'targets'
        global_metrics_evaluator.add_batch_of_examples(
            target=train_utils.unreplicate_and_get(
                e_batch[t_key])[e_batch_mask],
            output=train_utils.unreplicate_and_get(e_output)
            [e_batch_mask])
        del e_batch, e_output, e_batch_mask
    eval_global_metrics_summary = None
    if compute_global_metrics:
      eval_global_metrics_summary = (
          global_metrics_evaluator.compute_metrics(
              clear_annotations=True))
    eval_summary.update(
        train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            extra_eval_summary=eval_global_metrics_summary,
            writer=writer,
            prefix=val_name))
    del eval_metrics, eval_global_metrics_summary
    writer.flush()
    return eval_summary

  # If `global_metrics` are set in the config and we are the lead host
  compute_global_metrics = False
  if config.get('global_metrics', False) and lead_host:
    compute_global_metrics = True
  if compute_global_metrics:
    global_metrics_evaluator = sv_metric.TruvariGlobalEvaluator(
        config.global_metrics)

  eval_summary = None

  ################### EVALUATION #######################
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)
  eval_summary = evaluate_internal(train_state, 0, dataset.valid_iter,
                                   dataset.meta_data['num_eval_examples'],
                                   'SV_test')

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return eval_summary
