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

"""Main file for Audiovisual Masked Autoencoders."""

from absl import flags
from clu import metric_writers
from clu import platform
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.av_mae import registry
from scenic.projects.av_mae import trainer as avmae_trainer
from scenic.projects.av_mae import trainer_multimodal as avmae_multimodal_trainer
from scenic.projects.av_mae import transfer_trainer as avmae_transfer_trainer
from scenic.projects.av_mae import transfer_trainer_multimodal
from scenic.train_lib_deprecated import train_utils


FLAGS = flags.FLAGS
FINAL_CKPT_ARTIFACT_DESCRIPTION = 'Final checkpoint'

# Flax checkpointing is deprecated. This is a temporary fix.
flax.config.update('flax_use_orbax_checkpointing', False)


def get_trainer(trainer_name):
  """Returns the trainer to use."""
  if trainer_name == 'avmae_trainer':
    return avmae_trainer.train
  elif trainer_name == 'avmae_transfer_trainer':
    return avmae_transfer_trainer.train
  elif trainer_name == 'avmae_multimodal_trainer':
    return avmae_multimodal_trainer.train
  elif trainer_name == 'transfer_trainer_multimodal':
    return transfer_trainer_multimodal.train
  else:
    raise ValueError(f'Unsupported trainer: {trainer_name}')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for AV-MAE training."""
  model_cls = registry.get_model_cls(config.model_name)

  # Don't write to datatable, as it cannot handle large image summaries.
  del writer
  writer = metric_writers.create_default_writer(
      FLAGS.workdir, just_logging=jax.process_index() > 0, asynchronous=True,
      write_to_datatable=False)

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
  app.run(main)
