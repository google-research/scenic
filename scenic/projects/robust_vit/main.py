"""Main file for Robust ViT."""

from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.model_lib import models
from scenic.projects.robust_vit import model
from scenic.projects.robust_vit import robustness_evaluator as vit_robustness_evaluator
from scenic.projects.robust_vit import trainer
from scenic.projects.robust_vit import vqmixer
from scenic.train_lib import train_utils
import scenic.train_lib.trainers as general_trainer

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'robust_vit_multilabel_classification' or model_name == 'rob_vit_multilabel_classification':
    return model.RobViTMultiLabelClassificationModel
  elif model_name == 'robust_mixer_multilabel_classification':
    return vqmixer.VQMixerMultiLabelClassificationModel
  else:
    # To be compatible with saved models.
    if model_name == 'vit_multilabel_classification_dropout':
      model_name = 'vit_multilabel_classification'
    return models.get_model_cls(model_name)


def get_trainer(trainer_name: str) -> Callable[..., Any]:
  """Returns trainer given its name."""
  if trainer_name == 'vit_robustness_evaluator':
    return vit_robustness_evaluator.evaluate
  elif trainer_name == 'robust_trainer':
    return trainer.train
  else:
    return general_trainer.get_trainer(trainer_name)
  raise ValueError(f'Unsupported trainer: {trainer_name}.')


def update_config_for_evaluation(
    config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Updates the config for robustness evaluation."""
  if config.trainer_name != 'vit_robustness_evaluator':
    return config
  else:
    return vit_robustness_evaluator.update_config(config)


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the ViViT project."""
  config = update_config_for_evaluation(config)
  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  robust_train = get_trainer(config.trainer_name)

  robust_train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
