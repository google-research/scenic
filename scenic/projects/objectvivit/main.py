"""Main file for ObjectViViT."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app

from scenic.projects.objectvivit import model
from scenic.projects.objectvivit import trainer
# pylint: disable=unused-import
import scenic.projects.objectvivit.datasets
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def get_model_cls(model_name):
  """"Selects Vivit model type."""
  if model_name == 'vivit_classification':
    return model.ViViTModelWithObjects
  else:
    raise ValueError('Unrecognized model: {}'.format(model_name))


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the ViViT project."""
  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
