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

"""Main file for launching experiments."""

import pdb  # pylint: disable=unused-import

from absl import flags
import chex
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.ncr import classification_trainer
from scenic.projects.ncr import resnet as ncr_resnet
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def get_model_cls(model_name):
  if model_name == 'resnet':
    return ncr_resnet.ResNetNCRModel
  else:
    raise ValueError(f'Unknown model {model_name}')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the Scenic."""

  # Disable pmap if debugging
  if config.get('fake_pmap', False):
    fake_pmap = chex.fake_pmap()
    fake_pmap.start()
  else:
    fake_pmap = None

  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  classification_trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)

  if fake_pmap is not None:
    fake_pmap.stop()

if __name__ == '__main__':
  app.run(main=main)
