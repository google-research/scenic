"""Main file for Scenic."""

from absl import flags
from clu import metric_writers
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.adversarialtraining import classification_adversarialtraining_trainer
from scenic.projects.adversarialtraining.models import models
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the Scenic."""

  # Enable wrapping of all module calls in a named_call for easier profiling:
  nn.enable_named_call()

  model_cls = models.get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  classification_adversarialtraining_trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
