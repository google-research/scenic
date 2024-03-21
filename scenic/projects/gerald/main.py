"""Main file for GER experiments."""

from absl import flags
from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.gerald import ger_trainer
from scenic.projects.gerald import input_pipeline

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main entry point for GER training."""
  data_rng, rng = jax.random.split(rng)
  workdir_config = config.get('workdir')
  if workdir_config:
    workdir = workdir_config
  logging.info('Workdir is %s', workdir)

  return ger_trainer.train_and_evaluate(
      rng=rng,
      config=config,
      dataset=input_pipeline.get_dataset(config, data_rng),
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
