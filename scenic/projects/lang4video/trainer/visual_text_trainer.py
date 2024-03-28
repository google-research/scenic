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

"""Scenic trainer for visual-text matching."""

from collections.abc import Callable
from collections.abc import Mapping
import functools
from typing import Any
from typing import Optional

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import optax

from scenic.dataset_lib import dataset_utils
from scenic.projects.lang4video.model.image_text_model import ImageTextModel
from scenic.projects.lang4video.trainer import optimizers
from scenic.projects.lang4video.trainer.train_utils import axis_name_exists
from scenic.projects.lang4video.trainer.train_utils import clip_grads
from scenic.projects.lang4video.trainer.train_utils import compute_mask
from scenic.projects.lang4video.trainer.train_utils import get_epoch_steps
from scenic.projects.lang4video.trainer.train_utils import get_input_spec
from scenic.projects.lang4video.trainer.train_utils import init_encoder
from scenic.projects.lang4video.trainer.train_utils import is_video_input
from scenic.projects.lang4video.trainer.train_utils import NUM_DEVICES_AXIS_NAME
from scenic.train_lib import lr_schedules
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


def train_step(
    train_state: train_utils.TrainState,
    visual: jnp.ndarray,  # Shape: (N, F, H, W, C) if `is_video`, w/o F if not.
    text: jnp.ndarray,  # Shape: (N, L)
    mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
    *,
    model: ImageTextModel,
    is_video: bool,
    lr_fn: Callable[[int], float],
    config: ml_collections.ConfigDict,
    gather_scores: bool = True,
    debug: bool = False,
) -> tuple[train_utils.TrainState, Mapping[str, Any], Mapping[str, Any]]:
  """Runs a single step of evaluation."""
  encoder = model.flax_model

  new_rng, rng = jax.random.split(train_state.rng)

  if axis_name_exists(NUM_DEVICES_AXIS_NAME):
    rng = train_utils.bind_rng_to_host_device(
        rng, NUM_DEVICES_AXIS_NAME, bind_to='device')

  def loss_fn(
      params: Mapping[str, Any]
  ) -> tuple[float, tuple[Mapping[str, Any], jnp.ndarray]]:
    (encoded_visual, encoded_text), new_model_state_ = encoder.apply(
        variables={
            'params': params,
            **train_state.model_state,
        },
        mutable=train_state.model_state.keys(),
        method=model.flax_model.encode_video_and_text if is_video else None,
        rngs={'dropout': rng},
        **{'video' if is_video else 'image': visual},
        text=text,
        mask=mask,
        train=True,
        debug=debug)

    scores = encoder.compute_similarity(
        encoded_text,
        encoded_visual,
        all_gather_axis_name=(NUM_DEVICES_AXIS_NAME if gather_scores and
                              axis_name_exists(NUM_DEVICES_AXIS_NAME) else
                              None))

    loss_ = model.loss_function(scores).mean()

    return loss_, new_model_state_  # pytype: disable=bad-return-type  # jax-ndarray

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, new_model_state), grad = grad_fn(train_state.params)

  if axis_name_exists(NUM_DEVICES_AXIS_NAME):
    grad = jax.lax.pmean(grad, axis_name=NUM_DEVICES_AXIS_NAME)

  if max_grad_norm := config.get('max_grad_norm'):
    grad = clip_grads(grad, max_grad_norm)

  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,  # pytype: disable=attribute-error
                                                 train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      model_state=new_model_state,
      params=new_params,
      rng=new_rng)

  # We don't need to look at `batch_mask` because we drop the remainder.
  local_batch_size = len(visual)
  metrics = {'loss': (loss * local_batch_size, local_batch_size)}

  training_logs = {
      'temperature': 1 / jnp.exp(encoder.logit_scale),
      'step': train_state.global_step,
      'batch_size': local_batch_size * jax.device_count(),
      'lr': lr_fn(train_state.global_step),
  }

  return new_train_state, metrics, training_logs


def eval_step(
    train_state: train_utils.TrainState,
    visual: jnp.ndarray,  # Shape: (N, F, H, W, C) if `is_video`, w/o F if not.
    text: jnp.ndarray,  # Shape: (N, L)
    mask: Optional[jnp.ndarray],  # Shape: (N, L)
    batch_mask: jnp.ndarray,
    *,
    model: ImageTextModel,
    is_video: bool,
    gather_scores: bool = True,
    debug: bool = False,
) -> Mapping[str, Any]:
  """Runs a single step of evaluation."""
  encoder = model.flax_model

  encoded_visual, encoded_text = encoder.apply(
      variables={
          'params': train_state.params,
          **train_state.model_state,
      },
      method=model.flax_model.encode_video_and_text if is_video else None,
      **{'video' if is_video else 'image': visual},
      text=text,
      mask=mask,
      train=False,
      debug=debug)

  scores = encoder.compute_similarity(
      encoded_text,
      encoded_visual,
      all_gather_axis_name=(NUM_DEVICES_AXIS_NAME if gather_scores and
                            axis_name_exists(NUM_DEVICES_AXIS_NAME) else None))

  if gather_scores and axis_name_exists(NUM_DEVICES_AXIS_NAME):
    batch_mask = jax.lax.all_gather(batch_mask, NUM_DEVICES_AXIS_NAME)
    batch_mask = batch_mask.reshape(-1, batch_mask.shape[-1])

  batch_mask_sum = batch_mask.sum()

  metrics = {
      'loss': (model.loss_function(scores) * batch_mask, batch_mask_sum),
  }

  if not gather_scores and axis_name_exists(NUM_DEVICES_AXIS_NAME):
    # We need to gather the metrics then.
    actual_batch_size = jax.lax.psum(
        batch_mask_sum, axis_name=NUM_DEVICES_AXIS_NAME)
    for k, v in metrics.items():
      metrics[k] = (jax.lax.psum(v[0].sum(), axis_name=NUM_DEVICES_AXIS_NAME),
                    actual_batch_size)

  return metrics


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: type[ImageTextModel],
    dataset: dataset_utils.Dataset,
    workdir: Optional[str] = None,  # pylint: disable=unused-argument
    writer: metric_writers.MetricWriter,
) -> tuple[Mapping[str, float], Mapping[str, float]]:
  """Trains a model for visual-text matching."""
  input_spec = get_input_spec(
      dataset_meta_data=dataset.meta_data,
      dataset_configs=config.get('dataset_configs', {}),
      train=True)

  is_video = is_video_input(input_spec)

  lead_host = jax.process_index() == 0

  model = model_cls(config, dataset.meta_data)

  init_rng, rng = jax.random.split(rng)
  params, model_state = init_encoder(
      encoder=model.flax_model,
      input_spec=input_spec,
      method=model.flax_model.encode_video_and_text if is_video else None,
      config=config,
      rng=init_rng)

  lr_fn = lr_schedules.get_learning_rate_fn(config)
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  tx = optimizers.get_optimizer(optimizer_config, lr_fn, params=params)
  if optimizer_config.get('lookahead'):
    params = optax.LookaheadParams.init_synced(params)
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  chrono = train_utils.Chrono()

  train_rng, rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      tx=tx,
      opt_state=opt_state,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()})

  start_step = train_state.global_step

  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  chrono.load(train_state.metadata['chrono'])

  if start_step == 0 and (init_from := config.get('init_from')):
    init_checkpoint_path = init_from.get('checkpoint_path')
    if init_checkpoint_path:
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
      train_state = train_state.replace(
          params=restored_train_state.params,
          model_state=restored_train_state.model_state)
      del restored_train_state

  train_state = jax_utils.replicate(train_state)

  if (batch_size := config.batch_size) == 1:
    logging.warning('The batch size is 1, so then the NCE loss will be 0.')

  gather_scores = config.get('model', {}).get('gather_scores', True)

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          is_video=is_video,
          lr_fn=lr_fn,
          config=config,
          gather_scores=gather_scores,
          debug=config.get('debug_train'),
      ),
      axis_name=NUM_DEVICES_AXIS_NAME,
      donate_argnums=(0, 1, 2, 3),
  )

  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          model=model,
          is_video=is_video,
          gather_scores=gather_scores,
          debug=config.get('debug_eval'),
      ),
      axis_name=NUM_DEVICES_AXIS_NAME,
      donate_argnums=(1, 2, 3),
  )

  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  eval_steps = get_epoch_steps(config, dataset, split='eval')

  chrono.inform(
      first_step=start_step,
      total_steps=total_steps,
      global_bs=batch_size,
      steps_per_epoch=steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)

  if jax.process_index() == 0:
    async_manager = checkpoints.AsyncManager()  # pylint: disable=unused-variable
  else:
    async_manager = None  # pylint: disable=unused-variable

  # We use this object to report the progress in the work unit notes (`chrono`
  # also writes similar ones, but whatever), but also to time the eval and
  # checkpoint saving. It writes some metrics to the writer, while `chrono`
  # writes others.
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)

  def write_note(note: str) -> None:
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if lead_host and config.get('xprof', True):
    hooks.append(periodic_actions.Profile(logdir=workdir))

  train_summary, eval_summary = {}, {}
  metrics_list, training_logs_list = [], []

  write_note('Starting first step (and compilations)…')
  assert not isinstance(dataset.train_iter, dict)
  for step, batch in zip(
      range(start_step + 1, total_steps + 1), dataset.train_iter):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      if (text := batch.get('text_indices')) is None:  # pytype: disable=attribute-error
        text = batch['label']
      else:
        text = text[:, :, 0]

      mask = compute_mask(text, config)
      train_state, metrics, training_logs = train_step_pmapped(
          train_state,
          batch['inputs'],
          text,
          mask,
      )

      # This will accumulate metrics and training logs in the device memory up
      # to the point that we log them. This is no problem for small metrics but
      # may be a problem for large metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      metrics_list.append(metrics)
      training_logs_list.append(training_logs)

    for h in hooks:
      h(step)

    if ((step % config.get('log_summary_steps', 100) == 0) or (step == 1) or
        (step == total_steps) or (lead_host and chrono.warmup)):
      chrono.pause(wait_for=(metrics_list))

      if lead_host:
        chrono.tick(step, writer, write_note)

      train_summary = train_utils.log_train_summary(
          step=step,
          writer=writer,
          train_metrics=train_utils.unreplicate_and_get(metrics_list),
          extra_training_logs=train_utils.unreplicate_and_get(
              training_logs_list),
          key_separator='/',
          flush_writer=False)
      metrics_list, training_logs_list = [], []

      chrono.resume()

    if (config.get('eval_while_training', True) and
        ((step % config.get('log_eval_steps', steps_per_epoch) == 0) or
         (step == 1) or (step == total_steps))):
      chrono.pause(wait_for=train_state.params)

      with report_progress.timed('eval'):
        eval_metrics_all = []
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for _, eval_batch in zip(range(eval_steps), dataset.valid_iter):
          text = batch['label']
          mask = compute_mask(text, config)
          eval_metrics_batch = eval_step_pmapped(
              train_state,
              eval_batch['inputs'],
              text,
              mask,
          )
          eval_metrics_all.append(eval_metrics_batch)
        eval_summary = train_utils.log_eval_summary(
            step=step,
            writer=writer,
            eval_metrics=train_utils.unreplicate_and_get(eval_metrics_all),
            key_separator='/',
            flush_writer=False,
        )

      chrono.resume()

    if ((step % config.get('checkpoint_steps', steps_per_epoch) == 0 and
         step > 0) or (step == total_steps)) and config.get('checkpoint'):
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))

      with report_progress.timed('checkpoint'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          unreplicated_train_state = jax_utils.unreplicate(train_state)
          metadata = unreplicated_train_state.metadata
          metadata['chrono'] = chrono.save()
          unreplicated_train_state = unreplicated_train_state.replace(
              metadata=metadata)  # pytype: disable=attribute-error
          train_utils.save_checkpoint(
              workdir,
              unreplicated_train_state,
              overwrite=config.get('overwrite_checkpoint', False),
              # async_manager=async_manager,
          )
          del unreplicated_train_state

      chrono.resume()

  writer.flush()

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()

  return train_summary, eval_summary
