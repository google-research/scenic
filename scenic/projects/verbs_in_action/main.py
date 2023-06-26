"""Main file for launching trainings."""

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.verbs_in_action import clip4clip_model
from scenic.projects.verbs_in_action import tfrecord_dataset  # pylint: disable=unused-import
from scenic.projects.verbs_in_action import trainer
from scenic.train_lib import train_utils


FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """The main entry point, sets and runs the training loop."""
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  trainer.train_and_eval(
      rng=rng,
      config=config,
      model_cls=clip4clip_model.VideoAndTextModel,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
