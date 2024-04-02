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

"""Main file for Streaming DVC."""

import os
from typing import Any, Optional

from absl import flags
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import ml_collections

from scenic import app
from scenic.projects.streaming_dvc import evaluate
from scenic.projects.streaming_dvc import trainer
from scenic.projects.streaming_dvc.io import densecap_ops  # pylint: disable=unused-import
from scenic.projects.streaming_dvc.io import flexio  # pylint: disable=unused-import
from scenic.projects.streaming_dvc.io import ops  # pylint: disable=unused-import
from scenic.projects.streaming_dvc.modeling import model
from scenic.projects.streaming_dvc.modeling import streaming_model
from scenic.projects.streaming_dvc.modeling import vid2seq_model
from scenic.train_lib import train_utils

# replace with the path to your JAVA bin location
JRE_BIN_JAVA = path_to_jre_bin_java
FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'git':
    return model.CaptioningModel
  elif model_name == 'streaming_model':
    return streaming_model.StreamingCaptioningModel
  elif model_name == 'streaming_dense_model':
    return streaming_model.DenseStreamingCaptioningModel
  elif model_name in ['streaming_vid2seq', 'vid2seq']:
    return vid2seq_model.Vid2SeqModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def get_dataset(config: ml_collections.ConfigDict, data_rng: jnp.ndarray,
                dataset_service_address: Optional[str] = None):
  """Returns dataset given config."""
  return train_utils.get_dataset(
      config, data_rng, dataset_service_address=dataset_service_address)


def get_trainer_fn(config: ml_collections.ConfigDict):
  """Returns trainer function given config."""

  trainer_name = config.get('trainer', '')
  eval_only = config.get('eval_only', False) or trainer_name == 'evaluator'

  if eval_only:
    trainer_fn = evaluate.evaluate
  else:
    trainer_fn = trainer.train_and_evaluate

  return trainer_fn


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the Elcap project."""
  # Temporary fixing flax checkpoint migration issue.
  flax.config.update('flax_use_orbax_checkpointing', False)
  jave_jre = JRE_BIN_JAVA
  os.environ['JRE_BIN_JAVA'] = java_jre

  model_cls = get_model_cls(config.model.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  trainer_fn = get_trainer_fn(config)

  trainer_fn(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
