"""Main file for MatViT."""

from typing import Any

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.matvit import matvit
from scenic.projects.matvit import trainer
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'matvit_multilabel_classification':
    return matvit.MatViTMultiLabelClassificationModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the MatViT."""
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  trainer.train(
      rng=rng,
      config=config,
      model_cls=matvit.MatViTMultiLabelClassificationModel,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main)
