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

"""Main file for FastViT."""

from typing import Any

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.tasseo import classification_trainer as tasseo_classification_trainer
from scenic.projects.tasseo import duplex_vit
from scenic.projects.tasseo import duplex_vit_classification_trainer
from scenic.projects.tasseo import inference
from scenic.projects.tasseo import transfer_trainer
from scenic.projects.tasseo import vit
from scenic.projects.tasseo import xvit

# pylint: disable=unused-import
from scenic.projects.tasseo.datasets import abnormality_baseline_dataset
from scenic.projects.tasseo.datasets import chrmID_baseline_dataset
from scenic.projects.tasseo.datasets import chrmID_big_metaphase_context_dataset
from scenic.projects.tasseo.datasets import chrmID_dataset
from scenic.projects.tasseo.datasets import chrmID_metaphase_context_dataset
from scenic.projects.tasseo.datasets import longtail_baseline_dataset
from scenic.projects.tasseo.datasets import longtail_rhs_baseline_dataset
from scenic.projects.tasseo.datasets import metaphase_sexid_dataset
# pylint: enable=unused-import
from scenic.train_lib import train_utils
from scenic.train_lib import trainers

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'xvit_classification':
    return xvit.XViTClassificationModel
  elif model_name == 'vit_classification':
    return vit.ViTClassificationModel
  elif model_name == 'topological_vit_classification':
    return vit.TopologicalViTClassificationModel
  elif model_name == 'duplex_vit_classification':
    return duplex_vit.DuplexViTClassificationModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def get_trainer(trainer_name: str) -> Any:
  """Gets the trainer matching the given name."""
  if trainer_name == 'duplex_vit_classification_trainer':
    return duplex_vit_classification_trainer.train
  elif trainer_name == 'classification_trainer':
    return tasseo_classification_trainer.train
  elif trainer_name == 'transfer_trainer':
    return transfer_trainer.train
  elif trainer_name == 'inference':
    return inference.evaluate
  else:
    return trainers.get_trainer(trainer_name)


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the Tasseo."""
  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  get_trainer(config.trainer_name)(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main)
