"""Main file for FastViT."""

from typing import Any

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.svvit import classification_trainer as ag_trainer
from scenic.projects.svvit import vit
from scenic.projects.svvit import xvit
# pylint: disable=unused-import
from scenic.projects.svvit.datasets import pileup_coverage_dataset
from scenic.projects.svvit.datasets import pileup_window_dataset
# pylint: enable=unused-import
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'xvit_classification':
    return xvit.XViTClassificationModel
  elif model_name == 'vit_classification':
    return vit.ViTClassificationModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for SVViT."""
  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  ag_trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main)
