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

"""Main file for Scenic."""

from absl import flags
from absl import logging
from clu import metric_writers
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.model_lib import models
from scenic.train_lib import train_utils
from scenic.train_lib import trainers


FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter) -> None:
  """Main function for Scenic."""

  model_cls = models.get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)

  if config.checkpoint:
    # When restoring from a checkpoint, change the dataset seed to ensure that
    # the example order is new. With deterministic data, this ensures enough
    # randomization and in the future with deterministic data + random access,
    # we can feed the global step to the dataset loader to always continue
    # reading the rest of the data if we resume a job that was interrupted.
    checkpoint_path = checkpoints.latest_checkpoint(workdir)
    logging.info('CHECKPOINT PATH: %s', checkpoint_path)
    if checkpoint_path is not None:
      global_step = train_utils.checkpoint_path_step(checkpoint_path) or 0
      logging.info('Folding global_step %s into dataset seed.', global_step)
      data_rng = jax.random.fold_in(data_rng, global_step)

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  trainers.get_trainer(config.trainer_name)(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
