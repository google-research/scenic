"""Main file for launching LOCA trainings."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.loca import loca_dataset  # pylint: disable=unused-import
from scenic.projects.loca import ops  # pylint: disable=unused-import
from scenic.projects.loca import trainer
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main entry point for loca training."""
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  trainer.train(
      rng=rng,
      config=config,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
