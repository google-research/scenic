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
