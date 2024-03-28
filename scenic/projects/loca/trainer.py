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

"""LOCA Training Script."""

import copy
import functools
from typing import Any, Callable, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from scenic.projects.loca import utils
from scenic.projects.loca import vit
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]


def loca_train_step(
    train_state: utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    momentum_parameter_scheduler: Callable[[int], float],
    loss_fn: Any,
    metrics_fn: Any,
    config: ml_collections.ConfigDict,
) -> Tuple[utils.TrainState, Dict[str, Tuple[float, int]]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    momentum_parameter_scheduler: Momentum parameter scheduler for EMA update.
    loss_fn: The cross-entropy loss function.
    metrics_fn: Reports relative position loss and accuracy.
    config: Configurations of the experiment.

  Returns:
    The updated state of training.
  """
  # Some preparations.
  new_rng, dropout_rng, droptok_rng = jax.random.split(train_state.rng, num=3)
  dropout_rng = train_utils.bind_rng_to_host_device(
      dropout_rng, axis_name='batch', bind_to='device')
  droptok_rng = train_utils.bind_rng_to_host_device(
      droptok_rng, axis_name='batch', bind_to='device')
  step = train_state.global_step
  momentum_parameter = momentum_parameter_scheduler(step)
  n_pos = config.n_ref_positions  # Number of reference positions.
  bs = batch['reference'].shape[0]  # Per-device batch size.
  n_q_foc = config.dataset_configs.number_of_focal_queries
  batch = utils.prepare_input(batch, config)

  def training_loss_fn(params):
    # Step 1): Forward pass on the REFERENCE view.
    use_ema = config.apply_cluster_loss
    drop_moment = 'late' if config.apply_cluster_loss else 'early'
    _, r_feat_targets, r_patch_features, _ = flax_model.apply(
        {'params': train_state.ema_params if use_ema else params},
        batch['reference'],
        seqlen=config.reference_seqlen,
        seqlen_selection=config.reference_seqlen_selection,
        drop_moment=drop_moment,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})

    # Step 2): Forward pass on the QUERY views.
    use_pe = True if config.apply_cluster_loss else False
    #      2) a) Query with `random`-style.
    q_rand_loc_pred, q_rand_feat_pred, _, q_rand_idx_kept = flax_model.apply(
        {'params': params},
        batch['query0'],
        inputs_kv=r_patch_features,
        seqlen=config.query_max_seqlen,
        use_pe=use_pe,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})
    #      2) b) Queries with `focal`-style.
    q_foc_loc_pred, q_foc_feat_pred, _, _ = flax_model.apply(
        {'params': params},
        batch['queries'],
        inputs_kv=jnp.tile(r_patch_features, (n_q_foc, 1, 1)),
        use_pe=use_pe,
        train=True,
        rngs={'dropout': dropout_rng})
    #      2) c) Batch the `random` and `focal` queries together: for both
    #            predictions (`q_loc_pred`) and targets (`q_loc_targets`).
    #
    # q_loc_pred is position logits for all the patches of the queries.
    q_loc_pred = jnp.concatenate([
        q_rand_loc_pred.reshape(-1, n_pos),
        q_foc_loc_pred.reshape(-1, n_pos)], axis=0)
    q_rand_loc_targets = batch['query0_target_position'].reshape(bs, -1)
    # If tokens were dropped in the query0 (i.e. `random`-style query):
    if len(q_rand_idx_kept) < q_rand_loc_targets.shape[1]:
      # then drop the corresponding target positions.
      q_rand_loc_targets = jnp.take(q_rand_loc_targets, q_rand_idx_kept, axis=1)
    q_foc_loc_targets = batch['target_positions']
    q_loc_targets = jnp.concatenate([q_rand_loc_targets.reshape(-1),
                                     q_foc_loc_targets.reshape(-1)],
                                    axis=0)
    q_r_intersect = q_loc_targets != -1  # intersection of reference and queries
    # q_loc_targets are the position to predict for all the patches of the
    # queries.
    q_loc_targets = jax.nn.one_hot(q_loc_targets, n_pos)

    # Step 3): Position prediction loss.
    localization_loss = loss_fn(q_loc_pred, q_loc_targets, q_r_intersect)

    # Step 4): Patch cluster prediction loss.
    feature_loss = 0
    if config.apply_cluster_loss:
      k = r_feat_targets.shape[-1]  # Output dimension for feature pred loss.
      q_feat_pred = jnp.concatenate([
          q_rand_feat_pred, q_foc_feat_pred], axis=0) / config.model.temperature
      # Feature targets.
      r_feat_targets = nn.softmax(r_feat_targets / config.sharpening, axis=-1)
      # We adjust the targets with Optimal Transport to prevent collapse.
      r_feat_targets = utils.sinkhorn(r_feat_targets, distributed=True)
      # (bs*N) x k -> bs x N x k
      r_feat_targets = r_feat_targets.reshape(bs, -1, k)
      # Feature targets for the random query.
      q_rand_feat_targets = jnp.take_along_axis(
          r_feat_targets, jnp.expand_dims(q_rand_loc_targets, axis=-1), axis=1)
      q_rand_feat_targets = q_rand_feat_targets.reshape(-1, k)
      # Feature targets for the focal queries.
      r_feat_targets = jnp.tile(r_feat_targets, (n_q_foc, 1, 1))
      q_foc_feat_targets = jnp.take_along_axis(
          r_feat_targets, jnp.expand_dims(q_foc_loc_targets, axis=-1), axis=1)
      q_foc_feat_targets = q_foc_feat_targets.reshape(-1, k)
      # Concatenate the targets for the random and focal queries.
      q_feat_targets = jnp.concatenate([
          q_rand_feat_targets, q_foc_feat_targets], axis=0)
      feature_loss = loss_fn(q_feat_pred, q_feat_targets, q_r_intersect)
      # `me-max` regularization.
      avg_prediction = jnp.mean(nn.softmax(q_feat_pred, axis=-1), axis=0)
      avg_prediction = jax.lax.pmean(avg_prediction, axis_name='batch')
      feature_loss += jnp.sum(avg_prediction * jnp.log(avg_prediction))

    total_loss = localization_loss + feature_loss
    return total_loss, (
        {'label': q_loc_targets, 'batch_mask': q_r_intersect},
        q_loc_pred, feature_loss)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (total_loss, (batch, logits, feature_loss)), grad = compute_gradient_fn(
      train_state.params)
  metrics = metrics_fn(logits, batch)
  metrics.update(
      dict(total_loss=(total_loss, 1), feature_loss=(feature_loss, 1)))

  # Update the network parameters.
  grad = jax.lax.pmean(grad, axis_name='batch')
  if config.get('max_grad_norm', None) is not None:
    grad = optimizers.clip_grads(grad, config.max_grad_norm)
  new_train_state = train_state
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    # update the teacher weights
    new_ema_params = jax.tree_util.tree_map(
        lambda s, t: momentum_parameter * t + (1 - momentum_parameter) * s,
        new_params,
        train_state.ema_params,
    )

    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        global_step=step + 1,
        opt_state=new_opt_state,
        params=new_params,
        ema_params=new_ema_params,
        rng=new_rng)
  return new_train_state, metrics


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the utils.TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    dataset: The dataset that has train_iter and meta_data.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training.
  """
  lead_host = jax.process_index() == 0

  # Build the loss_fn, metrics, and flax_model.
  model = vit.ViTLOCAModel(config, dataset.meta_data)

  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, _, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)
  # Only one model function but two sets of parameters.
  ema_params = copy.deepcopy(params)

  # Get learning rate and ema temperature schedulers.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  momentum_parameter_scheduler = lr_schedules.compound_lr_scheduler(
      config.momentum_rate)

  # Create optimizer.
  weight_decay_mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
  tx = optax.inject_hyperparams(optax.adamw)(
      learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
      mask=weight_decay_mask,)
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  # Create chrono class to track and store training statistics and metadata.
  chrono = train_utils.Chrono()

  # Create the TrainState to track training state (i.e. params and optimizer).
  train_state = utils.TrainState(
      global_step=0, opt_state=opt_state, tx=tx, params=params,
      ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})

  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)
  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params, ema_params
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  # The function that performs one step of loca training.
  loca_train_step_pmapped = jax.pmap(
      functools.partial(
          loca_train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn(),
          momentum_parameter_scheduler=momentum_parameter_scheduler,
          config=config),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )

  train_metrics, train_summary = [], None
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  report_progress = periodic_actions.ReportProgress(num_train_steps=total_steps,
                                                    writer=writer)
  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)
  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))
  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  logging.info('Starting training loop at step %d.', start_step + 1)
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, tm = loca_train_step_pmapped(train_state, train_batch)
      train_metrics.append(tm)
    for h in hooks:
      h(step)
    ###################### LOG TRAIN SUMMARY ########################
    if (step % config.get('log_summary_steps') == 1) or (step == total_steps):
      chrono.pause()
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          writer=writer)
      chrono.resume()
      train_metrics = []
    ##################### CHECKPOINTING ###################
    if ((step % config.get('checkpoint_steps') == 1 and step > 1) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          utils.save_checkpoint(workdir, unrep_train_state)
          del unrep_train_state
      chrono.resume()  # Un-pause now.
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train summary after last step.
  return train_state, train_summary
