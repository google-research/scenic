"""Main script for PolyViT project."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.polyvit import model as polyvit_model
from scenic.projects.polyvit import train_utils as polyvit_train_utils
from scenic.projects.polyvit import trainer as polyvit_trainer

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the PolyViT project."""
  # Build the loss_fn, metrics, and flax_model.
  model_cls = polyvit_model.PolyVitModel
  data_rng, rng = jax.random.split(rng)
  dataset_dict = polyvit_train_utils.get_datasets(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  polyvit_trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset_dict=dataset_dict,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
