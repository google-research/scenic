"""Main file for ViViT."""

from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.vivit import model as vivit_model
from scenic.projects.vivit import trainer as vivit_trainer
from scenic.train_lib_deprecated import train_utils

FLAGS = flags.FLAGS


def get_trainer(trainer_name: str) -> Callable[..., Any]:
  """Returns trainer given its name."""
  if trainer_name == 'vivit_trainer':
    return vivit_trainer.train
  raise ValueError(f'Unsupported trainer: {trainer_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the ViViT project."""
  model_cls = vivit_model.get_model_cls(config.model_name)
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
