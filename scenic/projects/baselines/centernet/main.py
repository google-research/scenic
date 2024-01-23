"""Main file for CenterNet."""

from typing import Any

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.baselines.centernet import evaluate
from scenic.projects.baselines.centernet import input_pipeline
from scenic.projects.baselines.centernet import trainer
from scenic.projects.baselines.centernet.modeling import centernet
from scenic.projects.baselines.centernet.modeling import centernet2

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'centernet':
    return centernet.CenterNetModel
  elif model_name == 'centernet2':
    return centernet2.CenterNet2Model
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the CenterNet project."""
  model_cls = get_model_cls(config.model.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = input_pipeline.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  eval_only = config.get('eval_only', False)
  if eval_only:
    evaluate.evaluate(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)
  else:
    trainer.train_and_evaluate(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)


if __name__ == '__main__':
  app.run(main=main)
