"""Evaluation Script.

The code evaluate a trained model on ImageNet variants in P@1 and P@5.
"""
import functools
from typing import Any, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.robust_vit.datasets import imagenet_variants
from scenic.projects.robust_vit.train_lib import train_utils as robust_vit_train_utils
from scenic.projects.robust_vit.train_lib.optimizers_utils import get_partial_optimizer
from scenic.train_lib import classification_trainer
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
PyTree = Any

IMAGENET_C_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
]

IMAGENET_C_SEVERITIES = range(1, 6)

ALEXENET_AVERAGE_ERRORS = {
    'gaussian_noise': 0.886,
    'shot_noise': 0.894,
    'impulse_noise': 0.923,
    'defocus_blur': 0.82,
    'snow': 0.867,
    'glass_blur': 0.826,
    'motion_blur': 0.786,
    'zoom_blur': 0.798,
    'frost': 0.827,
    'fog': 0.819,
    'brightness': 0.565,
    'contrast': 0.853,
    'elastic_transform': 0.646,
    'pixelate': 0.718,
    'jpeg_compression': 0.607,
}


def _dummy_metrics_function(logits: jnp.ndarray, batch: Batch):
  """Dummy function returns the logits."""
  del batch
  return 0, logits


def custom_eval(logits: jnp.ndarray,
                multi_hot_targets: jnp.ndarray,
                weights: Optional[jnp.ndarray] = None,
                appearing_classes: Optional[jnp.ndarray] = None,
                class_mappings: Optional[jnp.ndarray] = None):
  """Customer evaluation function for computing the top-1 and top-5 precicison.

  Similar to weighted_topk_correctly_classified in model_utils but with
  adapation to evaluate the model robustness.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_target -1).
    appearing_classes: None or 1d array indicting the classes to evaluate.
    class_mappings: None or 1d np array of mapping in which each mapping is a
      list that maps an imagenet id to a list of imagenet ids. It is used in the
      objectnet where the mapping is one to many, e.g. 'still_camera':
      [732, 759] is mapped to two ImageNet classes.

  Returns:
    Pecision at 1, Precision at 5 and count of examples for the mini-batch.
  """
  assert logits.shape == multi_hot_targets.shape
  if class_mappings is not None:
    multi_hot_targets = np.array(multi_hot_targets)
    for class_mapping in class_mappings:
      multi_hot_targets[..., class_mapping] = np.max(
          multi_hot_targets[..., class_mapping], -1, keepdims=True)

  if appearing_classes is not None:
    logits = logits[..., appearing_classes]
    multi_hot_targets = multi_hot_targets[..., appearing_classes]

  top1_idx = jnp.argmax(logits, axis=-1)
  top1_correct = jnp.take_along_axis(
      multi_hot_targets, top1_idx[..., None], axis=-1)
  top1_correct = jnp.squeeze(top1_correct)

  num_classes = logits.shape[-1]
  topk_pred = jax.lax.top_k(logits, 5)[1]
  multi_hot_pred = jnp.sum(
      jax.nn.one_hot(topk_pred, num_classes=num_classes), axis=-2)
  topk_correct = jnp.any(
      multi_hot_pred * multi_hot_targets, axis=-1).astype(jnp.float32)

  if weights is not None:
    # Ignores the entries with all zero label for evaluation.
    weights *= multi_hot_targets.max(axis=-1)
    top1_correct = model_utils.apply_weights(top1_correct, weights)
    topk_correct = model_utils.apply_weights(topk_correct, weights)
    counts = model_utils.apply_weights(jnp.ones_like(top1_correct), weights)
  return top1_correct.astype(jnp.int32), topk_correct.astype(
      jnp.int32), counts.astype(jnp.int32)


def get_custom_optimizer(config):
  """Get Optimizer based on Config setup.

  Args:
    config: Configuration setup.

  Returns:
    Optimizer class.
  """
  if config.optimizer == 'adam_vitonly':
    opt_class = get_partial_optimizer(config)
  else:
    opt_class = optimizers.get_optimizer(config)
  return opt_class


def update_config(
    user_config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Loads the real config for eval according to the user config."""
  assert ('xid' in user_config
          and 'wid' in user_config), 'Need to specify Xmanager xid and wid.'
  (xid, wid) = user_config.xid, user_config.wid
  assert (isinstance(xid, int) and
          isinstance(wid, int)), 'xid/wid is not integer.'


  # overrides the values in user_config
  config.update(user_config)
  config.experiment_name = f'{config.experiment_name}_robustness'

  # uses the customer ckpt_path if provided.
  if config.get('ckpt_path', None) is not None:
    # overrides the model path using the path in config.path
    config.init_from.checkpoint_path = config.ckpt_path
  else:
    config.init_from.checkpoint_path = path

  # updates the image_size
  config.dataset_configs.image_size = config.get('image_size', 224)

  if not config.dataset_configs.get('pp_eval', None):
    # pp_eval does not exist, equals None or being empty.
    config.dataset_configs.pp_eval = 'default'
  else:
    # For this code, we need a fixed label name as "label" not "labels"
    config.dataset_configs.pp_eval = config.dataset_configs.pp_eval.replace(
        '"labels"', '"label"')
  return config


def evaluate(
    *, rng: jnp.ndarray, config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel], dataset: dataset_utils.Dataset,
    workdir: str, writer: metric_writers.MetricWriter
) -> Tuple[train_utils.TrainState, Optional[Dict[int, Dict[str, Any]]]]:
  """Main evalution loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  evaluation, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has eval_iter, meta_data. Here only one dataset is
      needed and this function will creates other imagenet variants.
    workdir: Directory for checkpointing (unused).
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), eval_summary which
      is a dict of metrics. These outputs are used for regression testing.
  """
  del workdir  # Saves no checkpoint in evaluation.
  assert 'init_from' in config, (
      'Need to specify the model check_point either'
      'by init_from.checkpoint_path or (init_from.xm.xid and init_from.xm.wid)')
  model = model_cls(config, dataset.meta_data)
  data_rng, rng = jax.random.split(rng)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  params, model_state, _, _ = robust_vit_train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(dataset.meta_data['input_shape'],
                   dataset.meta_data.get('input_dtype', jnp.float32))],
      config=config,
      rngs=init_rng)

  # Create optimizer. Not used.
  opt_class = get_custom_optimizer(config)
  optimizer = jax.jit(
      opt_class.create, backend='cpu')(
          params)

  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng)

  init_checkpoint_path = config.init_from.get('checkpoint_path')
  # we have already set the init_checkpoint_path in update_config(...)
  assert init_checkpoint_path, 'Need a checkpoint path.'

  del params  # Do not keep a copy of the initial params.

  eval_step_pmapped = jax.pmap(
      functools.partial(
          classification_trainer.eval_step,
          flax_model=model.flax_model,
          metrics_fn=_dummy_metrics_function,  # just needs the logics here
          debug=config.debug_eval),
      axis_name='batch',
      donate_argnums=(1,),
  )

  def eval_fn(train_state: train_utils.TrainState,
              dataset: dataset_utils.Dataset) -> Dict[str, Any]:
    eval_batch_size = config.get('eval_batch_size', 32)
    total_eval_steps = int(
        np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
    steps_per_eval = config.get('steps_per_eval') or total_eval_steps
    appearing_classes = dataset.meta_data.get('appearing_classes')
    class_mappings = dataset.meta_data.get('class_mappings', None)

    eval_metric_top1 = []
    eval_metric_top5 = []
    example_counts = []
    for _ in range(steps_per_eval):
      eval_batch = next(dataset.valid_iter)
      _, logits = eval_step_pmapped(train_state=train_state, batch=eval_batch)
      # TODO(czm): This part is for evaluate model with BERT auxiliary task.
      # try:
      #   if len(logits) >= 2:
      #     logits = logits[0]
      # except:
      #   pass
      top1_acc, top5_acc, example_cnt = custom_eval(
          logits,
          eval_batch['label'],
          eval_batch['batch_mask'],
          appearing_classes=appearing_classes,
          class_mappings=class_mappings)
      eval_metric_top1.append(top1_acc)
      eval_metric_top5.append(top5_acc)
      example_counts.append(example_cnt)

    eval_metric_top1 = jnp.array(eval_metric_top1)
    eval_metric_top5 = jnp.array(eval_metric_top5)
    total_num_samples = jnp.sum(jnp.array(example_counts))

    top1_accuracy = jnp.sum(eval_metric_top1) * 1.0 / total_num_samples
    top5_accuracy = jnp.sum(eval_metric_top5) * 1.0 / total_num_samples
    ds_name = dataset.meta_data['dataset_name']
    eval_summary = {
        f'robustness_top1_acc/{ds_name}': top1_accuracy,
        f'robustness_top5_acc/{ds_name}': top5_accuracy,
        f'robustness_total_num_samples/{ds_name}': total_num_samples
    }
    return eval_summary

  #####################    EVALUATION     ############################
  eval_summary = {}
  checkpoint_steps = robust_vit_train_utils.all_steps(init_checkpoint_path)
  if config.init_from.get('evaluate_all_checkpoints', False):
    logging.info('-----Using all checkpoints from %s', init_checkpoint_path)
  else:
    logging.info('-----Using the latest checkpoint from %s',
                 init_checkpoint_path)
    checkpoint_steps = [checkpoint_steps[-1]]  # only uses the latest

  for checkpoint_step in checkpoint_steps:
    logging.info('-----Load checkpoint from %s at step %d',
                 init_checkpoint_path, checkpoint_step)
    train_state, start_step = train_utils.restore_checkpoint(
        init_checkpoint_path, train_state, step=checkpoint_step)

    # CL 399189446
    optimizer_target = flax.core.unfreeze(train_state.optimizer.target)
    robust_vit_train_utils.dict_dfs_update_bias_scale(optimizer_target, '')
    optimizer = optimizer.replace(target=flax.core.freeze(optimizer_target))
    train_state = train_state.replace(optimizer=optimizer)

    # Replicates the optimzier, state, and rng.
    train_state = jax_utils.replicate(train_state)
    eval_summary[start_step] = {}
    config.dataset_name = 'imagenet_variants'
    all_datasets = imagenet_variants.DATASET_INFO.keys()
    all_datasets = list(
        filter(lambda x: not x.startswith('imagenet2012_corrupted'),
               all_datasets))
    for dataset_name in all_datasets:
      logging.info('----->>Start evaluation on %s', dataset_name)
      config.dataset_configs.update({'dataset_name': dataset_name})
      dataset = train_utils.get_dataset(config, data_rng)
      dataset.meta_data['dataset_name'] = dataset_name
      cur_eval_summary = eval_fn(train_state, dataset)
      writer.write_scalars(start_step, cur_eval_summary)
      writer.flush()
      eval_summary[start_step].update(cur_eval_summary)
      logging.info('----->>End evaluation on %s', dataset_name)

    # Evaluate on imagenet_c
    if not config.dataset_configs.get('exclude_imagenet_c', False):
      # For imagenet_c we report the macro-average on 15x5 runs.
      accuracy_per_corruption = {}
      mce_per_corruption = {}
      p_at_1_key = 'robustness_top1_acc/imagenet_c'
      logging.info('----->>Start evaluation on imagenet_c')
      for corruption in imagenet_variants.IMAGENET_C_CORRUPTIONS:
        local_list = []    # list to compute macro average per corruption
        for severity in imagenet_variants.IMAGENET_C_SEVERITIES:
          config.dataset_configs.update({
              'dataset_name': f'imagenet2012_corrupted/{corruption}_{severity}'
          })
          dataset = train_utils.get_dataset(config, data_rng)
          dataset.meta_data[
              'dataset_name'] = 'imagenet_c'  # same for aggregation
          local_list.append(eval_fn(train_state, dataset))
        # Computes the local average per corruption.
        accuracy_per_corruption[
            corruption] = robust_vit_train_utils.average_list_of_dicts(
                local_list)
        # Computes MCEs.
        cur_err = 1.0 - accuracy_per_corruption[corruption][p_at_1_key]
        alexnet_err = ALEXENET_AVERAGE_ERRORS[corruption]
        mce_per_corruption[corruption] = {'imagenet_c/mce': cur_err/alexnet_err}

      logging.info(
          '----->>Ends evaluation on imagenet_c %d sets',
          len(imagenet_variants.IMAGENET_C_CORRUPTIONS) *
          len(imagenet_variants.IMAGENET_C_SEVERITIES))

      # Computes and logs the macro-average.
      assert len(accuracy_per_corruption) == len(
          imagenet_variants.IMAGENET_C_CORRUPTIONS), 'Summary Mismatched.'
      imagenet_c_macro_precision = robust_vit_train_utils.average_list_of_dicts(
          accuracy_per_corruption.values())
      imagenet_c_macro_mce = robust_vit_train_utils.average_list_of_dicts(
          mce_per_corruption.values())
      writer.write_scalars(start_step, imagenet_c_macro_precision)
      writer.write_scalars(start_step, imagenet_c_macro_mce)
      eval_summary[start_step].update(imagenet_c_macro_precision)
      eval_summary[start_step].update(imagenet_c_macro_mce)
      writer.flush()

      # Logs precision and mce per corruption under the key 'imagenet_c'.
      for corruption, summary_dict in accuracy_per_corruption.items():
        new_dict = {
            f'imagenet_c/top1_acc/{corruption}':
                summary_dict[p_at_1_key]
        }
        writer.write_scalars(start_step, new_dict)
        eval_summary[start_step].update(new_dict)
        new_dict = {
            f'imagenet_c/mce/{corruption}':
                mce_per_corruption[corruption]['imagenet_c/mce']
        }
        writer.write_scalars(start_step, new_dict)
        eval_summary[start_step].update(new_dict)
        writer.flush()
  return train_state, eval_summary
