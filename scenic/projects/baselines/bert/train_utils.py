"""Utilities for BERT trainer."""

import functools
from typing import Any, Dict, Tuple, Optional, Mapping, Union, List

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
from scenic.train_lib.google.transfer import fewshot_utils
import scipy
import sklearn.metrics

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


def f1_score_with_invalid(target: np.ndarray,
                          prediction: np.ndarray) -> Dict[str, float]:
  """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.

  Args:
    target: Numpy array of targets, either 0 or 1 (binary label space).
    prediction: Numpy array of model predictions, any integer value.

  Returns:
    F1 score, where any prediction != 0 or 1 is counted as wrong.
  """
  # Get indices of invalid predictions:
  invalid_idx_mask = np.logical_and(prediction != 0, prediction != 1)
  # For any prediction != 0 or 1, set it to the opposite of what the target is:
  prediction[invalid_idx_mask] = 1 - target[invalid_idx_mask]
  return {'f1': sklearn.metrics.f1_score(target, prediction)}


def matthews_corrcoef(target: np.ndarray,
                      prediction: np.ndarray) -> Dict[str, float]:
  """Returns Matthews correlation coefficient (MCC)."""
  return {
      'matthews_corrcoef': sklearn.metrics.matthews_corrcoef(
          target, prediction)
  }


def pearson_corrcoef(target: np.ndarray,
                     prediction: np.ndarray) -> Dict[str, float]:
  """Returns Pearson correlation coefficient."""
  return {'pearson_corrcoef': scipy.stats.pearsonr(target, prediction)[0]}


class BERTGlobalEvaluator():
  """Evaluator used for BERT global metrics evaluation."""

  def __init__(self, global_metrics: List[str]):
    self.global_metrics = global_metrics
    self.batches = None
    self._num_examples_added = 0

  def add_batch_of_examples(self, target: np.ndarray, output: np.ndarray):
    """Add a batch of examples to the evaluator.

    Args:
      target: Target to be predicted as a Numpy array.
      output: Output from the model as a Numpy array.
    """
    self._num_examples_added += output.shape[0]
    if self.batches is None:
      self.batches = (target, output)
    else:  # Append targets and outputs for the new examples.
      self.batches = (np.append(self.batches[0], target, axis=0),
                      np.append(self.batches[1], output, axis=0))

  def compute_metrics(self,
                      clear_annotations: Optional[bool] = True
                     ) -> Dict[str, Any]:
    """Computes the relevant metrics for all added <target, output> pairs."""
    metrics = {}
    if 'f1' in self.global_metrics:
      # Used for QQP and MRPC tasks.
      prediction = np.argmax(self.batches[1], axis=-1)
      metrics.update(
          f1_score_with_invalid(target=self.batches[0], prediction=prediction))

    if 'matthews_corrcoef' in self.global_metrics:
      # Used for COLA task.
      prediction = np.argmax(self.batches[1], axis=-1)
      metrics.update(
          matthews_corrcoef(target=self.batches[0], prediction=prediction))

    if 'pearson_corrcoef' in self.global_metrics:
      # Used for STS-B task (which is a regression task).
      metrics.update(
          pearson_corrcoef(
              target=np.squeeze(self.batches[0]),
              prediction=np.squeeze(self.batches[1])))

    if clear_annotations:
      self.clear()
    return metrics

  def clear(self):
    self.batches = None
    self._num_examples_added = 0

  def __len__(self):
    return self._num_examples_added


def initialize_bert_model(
    *,
    model_def: nn.Module,
    input_spec: Dict[str, Union[Tuple[Tuple[int, ...], jnp.dtype],
                                Tuple[int, ...], None]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state of BERT.

  Args:
    model_def: Definition of a model.
    input_spec: A dictionary of arg name to a (shape, dtype) pair specifying the
      shape and dtype of the input arguments. If unspecified the dtype is
      float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """
  batch_size = (config.batch_size //
                jax.device_count()) if config.get('batch_size') else None
  dummy_input = {}
  for name, spec in input_spec.items():
    if spec is not None:
      in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
          spec, batch_size=batch_size)
      dummy_input[name] = jnp.zeros(in_st.shape, in_st.dtype)
    else:
      dummy_input[name] = None

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = model_def.init(
        rngs, dummy_input, train=False, debug=False).pop('params')
    # Set bias in the head to low value, such that loss is small initially.
    if config.get('init_head_bias', None) is not None:
      init_params = flax.core.unfreeze(init_params)
      init_params['output_projection'] = optimizers.tree_map_with_names(
          lambda p: jnp.full_like(p, config.init_head_bias),
          init_params['output_projection'],
          match_name_fn=lambda name: 'bias' in name)
      init_params = flax.core.freeze(init_params)
    return init_params, init_model_state

  if not isinstance(rngs, dict):
    rngs = {'params': rngs}
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)

  # Count gflops:
  count_flops = config.get('count_flops',
                           ml_collections.ConfigDict({'count_flops': True}))
  if count_flops:
    variables = {'params': init_params, **init_model_state}
    flax_model_apply_fn = functools.partial(
        model_def.apply, variables, train=False, debug=False, rngs=rngs)
    m = jax.xla_computation(flax_model_apply_fn)(dummy_input).as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)  # pylint: disable=protected-access
    flops = analysis['flops']
    if count_flops.get('fuse_multiply_add', True):
      flops = flops / 2
    gflops = flops / (10**9)
    logging.info('GFLOPs %0.3f for input spec: %s', gflops, input_spec)
  else:
    gflops = None

  return init_params, init_model_state, num_trainable_params, gflops


class BERTFewShotEvaluator:
  """Class for BERT few-shot evaluation."""

  # These are classificcation tasks from GLUE that are pretty common and
  # standard indicator of quality of models, used for sanity checking.
  SUPPORTED_FEWSHOT_TASK_NAMES = [
      'sst2', 'mnli_matched', 'mnli_mismatched', 'rte', 'qnli'
  ]

  def __init__(self, representation_fn, fewshot_config):
    self.rng_seed = fewshot_config.get('rng_seed', 42)
    self.shots = fewshot_config.shots
    self.l2_regs = fewshot_config.l2_regs
    self.local_batch_size = fewshot_config.batch_size // jax.process_count()
    self.repr_fn = representation_fn
    self.walk_first = fewshot_config.walk_first
    self._datasets = {}  # This will be our cache for lazy loading.

  def _get_dataset(self, config):
    """Lazy-loads given dataset."""
    if config.dataset_configs.task not in self.SUPPORTED_FEWSHOT_TASK_NAMES:
      raise ValueError('dataset_configs.task_name must be one of [{}].'.format(
          ', '.join(self.SUPPORTED_FEWSHOT_TASK_NAMES)))
    key = config.dataset_configs.task
    try:
      return self._datasets[key]
    except KeyError:
      data_rng = jax.random.PRNGKey(self.rng_seed)
      dataset = train_utils.get_dataset(config, data_rng)
      train_ds = dataset.train_iter
      num_train_samples = dataset.meta_data['num_train_examples']
      test_ds = dataset.valid_iter
      num_test_samples = dataset.meta_data['num_eval_examples']
      num_classes = dataset.meta_data['num_classes']
      return self._datasets.setdefault(
          key,
          (train_ds, test_ds, num_train_samples, num_test_samples, num_classes))

  def _get_repr(self, train_state, data, num_samples):
    """Compute representation for the whole dataset."""
    pre_logits_list = []
    labels_list = []
    total_steps = int(np.ceil(num_samples / self.local_batch_size))
    for _ in range(1, total_steps + 1):
      batch = next(data)
      pre_logits, labels, mask = self.repr_fn(
          train_state=train_state, batch=batch)
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

  def compute_fewshot_metrics(self, train_state, config):
    """Compute few-shot metrics on one dataset."""
    (train_ds, test_ds, num_train_samples, num_test_samples,
     num_classes) = self._get_dataset(config)
    task = config.dataset_configs.task
    logging.info('[fewshot][%s]: Precomputing train', task)
    repr_train, labels_train = self._get_repr(train_state, train_ds,
                                              num_train_samples)
    logging.info('[fewshot][%s]: Precomputing test', task)
    repr_test, labels_test = self._get_repr(train_state, test_ds,
                                            num_test_samples)
    logging.info('[fewshot][%s]: solving systems', task)

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
        acc = fewshot_utils._fewshot_acc_fn(  # pylint: disable=protected-access
            x, y, repr_test, labels_test, l2_reg, num_classes)
        results[shots, l2_reg] = np.array(acc)
    return results

  def run_all(self, train_state, datasets):
    """Compute summary over all `datasets` that comes from config."""
    results = {}
    for cfg in datasets:
      results[cfg.dataset_configs.task] = self.compute_fewshot_metrics(
          train_state, cfg)

    # Now also figure out the regularization parameter that works best across
    # all datasets, per-shot. Similar to ATARI benchmark requiring one single
    # hyper-param across tasks, or BiT-HyperRule defining one clear thing.
    # Avoids over-fitting to a single task by selecting on test there, while
    # also avoiding the need to do cross-validation runs for each task.
    best_l2 = {}
    for shots in self.shots:
      reg_ranks = []
      for _, res in results.items():
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
      scalars[f'z/best_l2_for_{shots}shot_eval'] = l2

      for dataset_name, result in results.items():
        scalars[f'z/{dataset_name}_{shots}shot'] = result[shots, l2]

    # And a highlight, if desired:
    if self.walk_first:
      dataset_name, shots = self.walk_first
      l2 = best_l2[shots]
      highlight_value = results[dataset_name][shots, l2]
      scalars[f'a/{dataset_name}_{shots}shot'] = highlight_value

    writer.write_scalars(step, scalars)
