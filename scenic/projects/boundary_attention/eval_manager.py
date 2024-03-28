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

"""An eval manager.

It can be run either from a standalone eval binary or as part of a periodic
evaluation in a train loop.
"""

import contextlib
import functools
from typing import ContextManager, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.boundary_attention.helpers import viz_utils
from scenic.projects.boundary_attention.types import ArrayDict, MetricFn  # pylint: disable=g-multiple-import, g-importing-member
from scenic.train_lib import train_utils as scenic_train_utils


def eval_step(
    train_state: scenic_train_utils.TrainState,
    batch: ArrayDict,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
) -> Tuple[Dict[str, jnp.ndarray], ArrayDict]:
  """Runs a single step of evaluation."""

  variables = {'params': train_state.params, **train_state.model_state}

  rng = train_state.rng

  params_rng, codebook_rng, _ = jax.random.split(rng, 3)
  # Bind the rng to the host/device we are on.
  params_rng = scenic_train_utils.bind_rng_to_host_device(
      params_rng, axis_name='batch', bind_to='device'
  )

  rngs = {'params': params_rng, 'codebook': codebook_rng}

  # Run model
  model_outputs = flax_model.apply(
      variables, batch['image'], train=False, rngs=rngs
  )

  # Metrics
  metrics = metrics_fn(model_outputs, batch)

  return metrics, model_outputs


_SPARSE_BATCHES_FOR_VISUALIZATION = list(range(4))


class EvalManager:
  """Manages evaluations, deciding which to run at any given step."""

  def __init__(
      self,
      model,
      config: ml_collections.ConfigDict,
      rng: jax.Array,
      report_progress: Optional[periodic_actions.ReportProgress] = None,
  ):
    self.model = model
    self.config = config
    self.rng = rng
    self.report_progress = report_progress

    self.last_eval_step = -1

  def _evaluate(
      self,
      train_state: scenic_train_utils.TrainState,
      step: int,
      dataset: dataset_utils.Dataset,
      writer: metric_writers.MetricWriter,
  ):
    """Performs a single evaluation pass over the dataset for the given step."""

    valid_iter = dataset.valid_iter
    num_valid_ex = dataset.meta_data['num_eval_examples']
    # split = self.config.dataset.get('split', 'validation')
    split = 'test'

    if not isinstance(valid_iter, dict):  # Only on validation set.
      valid_iter = {'valid': (valid_iter, True)}
      num_valid_ex = {'valid': num_valid_ex}

    for val_name, (val_iter, gen_viz) in valid_iter.items():
      eval_step_pmapped = jax.pmap(
          functools.partial(
              eval_step,
              flax_model=self.model.flax_model,
              metrics_fn=self.model.get_metrics_fn(split),
          ),
          axis_name='batch',
      )

      num_ex = num_valid_ex[val_name]
      logging.info('Dataset %s has %d of examples', val_name, num_ex)
      # Ceil rounding such that we include the last incomplete batch.
      eval_batch_size = self.config.get(
          'eval_batch_size', self.config.batch_size
      )
      total_eval_steps = int(np.ceil(num_ex / eval_batch_size))
      steps_per_eval = self.config.get('steps_per_eval') or total_eval_steps
      if steps_per_eval > total_eval_steps:
        logging.warning(
            'Requested %d eval steps, but iterating through the full '
            'validation set only takes %d steps. Performing %d eval steps.',
            steps_per_eval,
            total_eval_steps,
            total_eval_steps,
        )
        steps_per_eval = total_eval_steps
      eval_metrics = []
      additional_summary = None
      for i in range(steps_per_eval):
        eval_batch = next(val_iter)

        e_metrics, model_outputs = eval_step_pmapped(
            batch=eval_batch, train_state=train_state
        )

        ################### Visualization ###################
        if (
            split == 'test'
            and gen_viz
            and i in _SPARSE_BATCHES_FOR_VISUALIZATION
            and self.config.get('visualize', False)
        ):
          img_val_name = f'{val_name}_batch{i}'
          write_images = viz_utils.get_viz_dict_from_batch(
              eval_batch, model_outputs, self.model, img_val_name
          )
          write_images = jax.tree_util.tree_map(
              scenic_train_utils.unreplicate_and_get, write_images
          )
          writer.write_images(step, write_images)

        ################### Save evaluation metrics ###################
        eval_metrics.append(scenic_train_utils.unreplicate_and_get(e_metrics))
        if i % 10 == 0 or i == steps_per_eval - 1:
          logging.info('Completed eval step %d of %d.', i + 1, steps_per_eval)

        #########################################################

      prefix = f'eval/{val_name}'
      scenic_train_utils.log_eval_summary(
          step=step,
          eval_metrics=eval_metrics,
          extra_eval_summary=additional_summary,
          writer=writer,
          prefix=prefix,
      )

      writer.flush()

  def _eval_stage_context(self, name: str) -> ContextManager[None]:
    """A context manager for each state of eval."""
    if self.report_progress:
      return self.report_progress.timed(name)
    return contextlib.nullcontext()

  def run_one_eval(
      self,
      train_state: scenic_train_utils.TrainState,
      step: int,
      dataset: dataset_utils.Dataset,
      writer: metric_writers.MetricWriter,
      is_final: bool,
  ):
    """Runs evaluations against a single train_state/step.

    Args:
      train_state: The train state being evaluated.
      step: The global step corresponding to the `train_state`.
      dataset: The dataset to compute metrics on. Note that fewshot and linear
        probe evals create their own datasets independent of this.
      writer: A metrics writer.
      is_final: Whether this is the final step being evaluated, so evals should
        be run even if they wouldn't normally run at this step.
    """
    log_eval_steps = self.config.get('log_eval_steps')
    if not log_eval_steps:
      raise ValueError("'log_eval_steps' should be specified in the config.")
    if (
        self.last_eval_step < 0
        or step >= self.last_eval_step + log_eval_steps
        or is_final
    ):
      logging.info('Running eval at step %d.', step)
      with self._eval_stage_context('eval'):
        # Sync model state across replicas.
        train_state = scenic_train_utils.sync_model_state_across_replicas(
            train_state
        )
        self._evaluate(train_state, step, dataset, writer)
        writer.flush()
        self.last_eval_step = step

    logging.info('Completed all evaluations for step %d,', step)
