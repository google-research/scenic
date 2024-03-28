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

"""Main file for GIT."""

import os
from typing import Optional

from absl import flags
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.pixel_llm import evaluate
from scenic.projects.pixel_llm import trainer
from scenic.projects.pixel_llm.io import flexio as custom_flexio  # pylint: disable=unused-import
from scenic.projects.pixel_llm.io import ops as pixel_llm_ops  # pylint: disable=unused-import
from scenic.projects.pixel_llm.modeling import pixel_llm as pixel_llm_model
from scenic.train_lib import train_utils


FLAGS = flags.FLAGS


def get_dataset(
    config: ml_collections.ConfigDict,
    data_rng: jnp.ndarray,
    dataset_service_address: Optional[str],
):
  """Returns dataset given config."""
  return train_utils.get_dataset(
      config,
      data_rng,
      dataset_service_address=dataset_service_address,
  )


def get_trainer_fn(config: ml_collections.ConfigDict):
  """Returns trainer function given config."""

  trainer_name = config.get('trainer', '')
  eval_only = config.get('eval_only', False) or trainer_name == 'evaluator'

  kwargs = {}
  if eval_only:
    trainer_fn = evaluate.evaluate
  else:
    trainer_fn = trainer.train_and_evaluate

  return trainer_fn, kwargs


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
):
  """Main function for the PixelLLM project."""
  # Temporary fixing flax checkpoint migration issue.
  flax.config.update('flax_use_orbax_checkpointing', False)

  trainer_fn, kwargs = get_trainer_fn(config)
  model_cls = pixel_llm_model.PixelLlmModel

  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config,
      data_rng,
      dataset_service_address=FLAGS.dataset_service_address,
  )

  trainer_fn(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer,
      **kwargs,
  )


if __name__ == '__main__':
  app.run(main=main)
