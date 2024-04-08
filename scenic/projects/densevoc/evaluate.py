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

"""Evaluation script for Dense VOC."""

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
from scenic.dataset_lib import dataset_utils
from scenic.projects.densevoc import evaluation_utils
from scenic.train_lib import train_utils


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

  if config.get('xid') and config.get('wid'):
    train_config, checkpoint_path = xm_utils.get_info_from_xmanager(
        config.xid, config.wid)
    train_config = evaluation_utils.override_train_model_config(
        train_config, config)
    model = model_cls(train_config, dataset.meta_data)
  else:
    model = model_cls(config, dataset.meta_data)
    checkpoint_path = config.weights

  inference_on_video = config.get('inference_on_video', False)
  checkpoint_data = checkpoints.restore_checkpoint(checkpoint_path, None)
  params = checkpoint_data['params']
  train_state = train_utils.TrainState(
      global_step=0,
      params=flax.core.FrozenDict(params),
      model_state=flax.core.FrozenDict({}),
      rng=rng)
  train_state = jax_utils.replicate(train_state)
  del checkpoint_data, params

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
    train_state = train_utils.sync_model_state_across_replicas(train_state)
    if inference_on_video:
      eval_results, eval_metrics = evaluation_utils.inference_on_video_dataset(
          model,
          train_state,
          dataset,
          config=config,
          eval_batch_size=eval_batch_size,
          is_host=is_host,
          save_dir=workdir,
      )
    else:
      eval_results, eval_metrics = evaluation_utils.inference_on_image_dataset(
          model,
          train_state,
          dataset,
          config=config,
          eval_batch_size=eval_batch_size,
          is_host=is_host,
          save_dir=workdir,
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
