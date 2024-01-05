"""Main file for launching UnLoc training jobs."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.unloc import model
from scenic.projects.unloc import single_task_trainer
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def get_trainer(trainer_name: str):
  """Returns trainer given its name."""
  if trainer_name == 'single_task_trainer':
    return single_task_trainer.train
  raise ValueError(f'Unsupported trainer: {trainer_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the UnLoc project."""
  model_cls = model.MODELS[config.model_name]
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
