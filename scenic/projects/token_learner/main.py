"""Main file for TokenLearner."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.token_learner import model
from scenic.projects.vivit import trainer as vivit_trainer
from scenic.train_lib_deprecated import train_utils
from scenic.train_lib_deprecated import trainers

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the TokenLearner."""
  model_cls = model.get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  if config.trainer_name == 'vivit_trainer':
    # ViViT trainer is not in the central Scenic registry for trainers.
    trainer = vivit_trainer.train
  else:
    trainer = trainers.get_trainer(config.trainer_name)

  trainer(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main)
