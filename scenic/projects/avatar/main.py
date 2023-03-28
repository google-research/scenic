"""Main file for training token generation models."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.avatar import generation_trainer
from scenic.projects.avatar import models
from scenic.train_lib_deprecated import train_utils


FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """The main entry point, sets and runs the training loop."""
  model_cls = models.Seq2SeqModel
  data_rng, rng = jax.random.split(rng)


  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  generation_trainer.train_and_eval(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
