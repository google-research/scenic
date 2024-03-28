# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script for Boundary Attention project."""

from absl import flags
from absl import logging
import chex
from clu import metric_writers
from clu import platform
from flax import linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app as scenic_app
from scenic.projects.boundary_attention import trainer
from scenic.projects.boundary_attention.dataset_lib import dataloader
from scenic.projects.boundary_attention.models import all_models

flags.DEFINE_string('dataset_dir', '', 'Dataset directory.')
flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path.')
flags.DEFINE_integer('checkpoint_step', -1, 'Checkpoint step.')
flags.DEFINE_string('weights_path', '', 'Pretrained weights path.')

FLAGS = flags.FLAGS
FINAL_CKPT_ARTIFACT_DESCRIPTION = 'Final checkpoint'


def collapse_str_to_int(workdir):
  return sum([ord(s) for s in workdir])


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the Boundary Attention."""

  # Update config if flags defined
  if len(FLAGS.dataset_dir) > 0:  # pylint: disable=g-explicit-length-test
    config.dataset.dataset_dir = FLAGS.dataset_dir
  if len(FLAGS.checkpoint_path) > 0:  # pylint: disable=g-explicit-length-test
    config.init_from.checkpoint_path = FLAGS.checkpoint_path
  if FLAGS.checkpoint_step != -1:
    config.init_from.checkpoint_step = FLAGS.checkpoint_step
  if len(FLAGS.weights_path) > 0:  # pylint: disable=g-explicit-length-test
    config.init_from.params_path = FLAGS.weights_path

  # Update learning rate to take into account the number of devices
  num_devices = jax.device_count()
  config.num_devices = num_devices
  config.lr_configs.base_learning_rate = (config.lr_configs.base_learning_rate *
                                          (num_devices / 2))
  config.lr_configs.end_learning_rate = (config.lr_configs.end_learning_rate *
                                         (num_devices / 2))
  logging.info('num_devices: %d', num_devices)

  # Build the loss_fn, metrics, and flax_model.
  model_cls = all_models.get_model_cls(config.model.name)
  data_rng, rng = jax.random.split(rng)
  dataset_shuffle = collapse_str_to_int(workdir)
  dataset = dataloader.get_dataloader(
      config,
      data_rng,
      dataset_service_address=FLAGS.dataset_service_address,
      dataset_shuffle=dataset_shuffle,
  )
  if config.disable_pmap_and_jit:
    chex.fake_pmap_and_jit().start()
  nn.enable_named_call()
  trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)

  # Log final checkpoint path as XManager artifact to tell parallel jobs that
  # training is done:
  if jax.process_index() == 0:
    # XManager overwrites artifacts with identical content even if their
    # description is different. As a workaround, we prepend "TRAIN", so that
    # the path can be distinguished from the "Last evaluated checkpoint"
    # artifact written by the evaluator.
    # TODO(b/210825478): Remove prepended string.
    artifact = 'TRAIN' + checkpoints.latest_checkpoint(workdir)
    platform.work_unit().create_artifact(platform.ArtifactType.FILE, artifact,
                                         FINAL_CKPT_ARTIFACT_DESCRIPTION)


if __name__ == '__main__':
  scenic_app.run(main=main)
