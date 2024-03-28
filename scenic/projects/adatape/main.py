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

"""Main file for AdaTape."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.adatape.adatape_vit import adatape_classify_trainer
from scenic.projects.adatape.adatape_vit import adatape_trainer
from scenic.projects.adatape.adatape_vit import adatape_vit
from scenic.projects.adatape.dataset import parity_dataset  # pylint: disable=unused-import
from scenic.train_lib import train_utils
from scenic.train_lib import trainers

FLAGS = flags.FLAGS


def get_model_cls(model_name: str):
  """Get the model class for the AdaTape project."""
  if model_name == 'adatape' or model_name == 'adatape-parity':
    return adatape_vit.AdaTapeMultiLabelClassificationModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def get_trainer(trainer_name):
  if trainer_name == 'adatape_trainer':
    return adatape_trainer.train
  elif trainer_name == 'adatape_classify_trainer':
    return adatape_classify_trainer.train
  else:
    return trainers.get_trainer(trainer_name)


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the AdaTape project."""
  # Build the loss_fn, metrics, and flax_model.
  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  trainer = get_trainer(config.trainer_name)
  trainer(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
