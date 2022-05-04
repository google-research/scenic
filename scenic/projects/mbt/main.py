r"""Main script for training MBT models."""

from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.mbt import model
from scenic.projects.mbt import trainer
from scenic.train_lib_deprecated import train_utils

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Callable[..., Any]:
  """Returns model class given its name."""
  if model_name == 'mbt_multilabel_classification':
    return model.MBTMultilabelClassificationModel
  elif model_name == 'mbt_classification':
    return model.MBTClassificationModel
  elif model_name == 'mbt_multihead_classification':
    return model.MBTMultiHeadClassificationModel
  raise ValueError(f'Unrecognized model: {model_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the MBT project."""
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
