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

"""Generic entry point for Python application in Scenic.

This provides run() which performs some initialization and then calls the
provided main with a JAX PRNGKey, the ConfigDict, the working directory
and a CLU MetricWriter.
We expect each scenic project to have its own main.py. It's very short but
makes it easier to maintain scenic as the number of projects grows.

Usage in your main.py:
  from scenic import app

  def main(rng: jnp.ndarray,
           config: ml_collections.ConfigDict,
           workdir: str,
           writer: metric_writers.MetricWriter):
    # Call the library that trains your model.

  if __name__ == '__main__':
    app.run(main)
"""
import functools
import os

from absl import app
from absl import flags
from absl import logging

from clu import metric_writers
from clu import platform
import flax
import flax.linen as nn
import jax
from ml_collections import config_flags
import tensorflow as tf

FLAGS = flags.FLAGS

# These are general flags that are used across most of scenic projects. These
# flags can be accessed via `flags.FLAGS.<flag_name>` and projects can also
# define their own flags in their `main.py`.
config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=False)
flags.DEFINE_string('workdir', None, 'Work unit directory.')
flags.DEFINE_string('dataset_service_address', None,
                    'Address of the tf.data service')
flags.mark_flags_as_required(['config', 'workdir'])

flax.config.update('flax_use_orbax_checkpointing', False)


def run(main):
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(functools.partial(_run_main, main=main))


def _run_main(argv, *, main):
  """Runs the `main` method after some initial setup."""
  del argv
  # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  config = FLAGS.config
  workdir = FLAGS.workdir
  if 'workdir_suffix' in config:
    workdir = os.path.join(workdir, config.workdir_suffix)

  # Enable wrapping of all module calls in a named_call for easier profiling:
  nn.enable_named_call()

  if FLAGS.jax_backend_target:
    logging.info('Using JAX backend target %s', FLAGS.jax_backend_target)
    jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else
                       FLAGS.jax_xla_backend)
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())

  # Add a note so that we can tell which task is which JAX host.
  # (task 0 is not guaranteed to be the host 0)
  platform.work_unit().set_task_status(
      f'host_id: {jax.process_index()}, host_count: {jax.process_count()}')
  if jax.process_index() == 0:
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         workdir, 'Workdir')

  rng = jax.random.PRNGKey(config.rng_seed)
  logging.info('RNG: %s', rng)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0, asynchronous=True)

  main(rng=rng, config=config, workdir=workdir, writer=writer)
