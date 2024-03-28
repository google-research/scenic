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

"""Evaluation binary for UnLoc models."""

import functools
from typing import Any, Dict, Tuple

from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.unloc import eval_utils as unloc_eval_utils
from scenic.projects.unloc import model as unloc_model
from scenic.projects.unloc import train_utils as unloc_train_utils
from scenic.train_lib import train_utils


def init_model(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState) -> Tuple[train_utils.TrainState, int]:
  """Initializes the UnLoc model."""

  checkpoint_dir = None
  checkpoint_path = config.init_from.get('checkpoint_path')
  if checkpoint_path is not None:
    checkpoint_dir = checkpoint_path
  train_state, step = train_utils.restore_checkpoint(
      checkpoint_dir, train_state, step=config.get('checkpoint_step')
  )
  return train_state, step


def evaluate(rng: jnp.ndarray, config: ml_collections.ConfigDict,
             writer: metric_writers.MetricWriter) -> Dict[str, Any]:
  """Evaluate an UnLoc model.

  This function runs a pretrained model on the test split of the specified
  dataset, and then evaluates the model.

  Args:
    rng: JAX prng key.
    config: Configuration of the model under evaluation.
    writer: CLU metrics writer instance.

  Returns:
    eval_summary: Dictionary with the evaluation summary
  """
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(config, data_rng)

  model = unloc_model.MODELS[config.model_name](
      config, dataset_meta_data=dataset.meta_data
  )

  rng, init_rng = jax.random.split(rng)
  params, model_state, _, _ = unloc_train_utils.initialize_model_with_pytree(
      model_def=model.flax_model,
      input_spec={
          'inputs': unloc_train_utils.create_input_spec(
              dataset.meta_data['input_shape'], dataset.meta_data['input_dtype']
          )
      },
      config=config,
      rngs=init_rng,
  )

  _, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=None,
      tx=None,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata=None)
  train_state, step = init_model(config, train_state)
  train_state = jax_utils.replicate(train_state)
  del params

  def action_segmentation_eval():
    test_step_pmapped = jax.pmap(
        functools.partial(
            unloc_eval_utils.action_segmentation_test_step,
            dataset=config.dataset_configs.get('name', ''),
            flax_model=model.flax_model,
            n_clips=config.get('multicrop_clips_per_device', 2),
            num_prompts=config.dataset_configs.get('num_prompts', 1),
            prompt_index=config.dataset_configs.get('prompt_index', None),
            debug=False,
        ),
        axis_name='batch',
    )
    return unloc_eval_utils.run_action_segmentation_test_steps_and_save_eval_summary(
        config, step, dataset, test_step_pmapped, train_state, writer
    )

  def tal_eval():
    test_step_pmapped = jax.pmap(
        functools.partial(
            unloc_eval_utils.temporal_localization_test_step,
            dataset='',
            task='temporal_localization',
            flax_model=model.flax_model,
            num_prompts=config.dataset_configs.get('num_prompts', 1),
            output_per_class_displacements=config.get(
                'output_per_class_displacements', True
            ),
            debug=False,
        ),
        axis_name='batch',
    )
    return unloc_eval_utils.run_temporal_localization_test_steps_and_save_eval_summary(
        config, step, dataset, test_step_pmapped, train_state, writer
    )

  def highlight_detection_eval():
    test_step_pmapped = jax.pmap(
        functools.partial(
            unloc_eval_utils.temporal_localization_test_step,
            dataset='',
            task='highlight_detection',
            flax_model=model.flax_model,
            num_prompts=config.dataset_configs.get('num_prompts', 1),
            output_per_class_displacements=config.get(
                'output_per_class_displacements', True
            ),
            debug=False,
        ),
        axis_name='batch',
    )
    return unloc_eval_utils.run_temporal_localization_test_steps_and_save_eval_summary(
        config, step, dataset, test_step_pmapped, train_state, writer
    )

  def moment_retrieval_eval():
    test_step_pmapped = jax.pmap(
        functools.partial(
            unloc_eval_utils.moment_retrieval_test_step,
            dataset=config.dataset_configs.get('name', ''),
            flax_model=model.flax_model,
            debug=False,
        ),
        axis_name='batch',
    )
    return (
        unloc_eval_utils.run_moment_retrieval_test_steps_and_save_eval_summary(
            config, step, dataset, test_step_pmapped, train_state, writer
        )
    )

  return {
      'action_segmentation': action_segmentation_eval,
      'highlight_detection': highlight_detection_eval,
      'moment_retrieval': moment_retrieval_eval,
      'temporal_localization': tal_eval,
  }[config.dataset_configs.task]()


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for offline evaluation."""
  del workdir
  evaluate(rng=rng, config=config, writer=writer)


if __name__ == '__main__':
  app.run(main=main)
