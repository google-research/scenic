"""Main file for OWL-ViT training."""

from absl import flags
from absl import logging
from clu import metric_writers
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit import trainer
from scenic.projects.owl_vit.preprocessing import input_pipeline  # pylint: disable=unused-import.
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main funtion for OWL-ViT training."""

  data_rng, rng = jax.random.split(rng)

  if config.checkpoint:
    # When restoring from a checkpoint, change the dataset seed to ensure that
    # the example order is new:
    train_state = checkpoints.restore_checkpoint(workdir, target=None)
    if train_state is not None:
      global_step = train_state.get('global_step', 0)
      logging.info('Folding global_step %s into dataset seed.', global_step)
      data_rng = jax.random.fold_in(data_rng, global_step)

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  trainer.train(
      rng=rng,
      config=config,
      model_cls=models.TextZeroShotDetectionModel,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
