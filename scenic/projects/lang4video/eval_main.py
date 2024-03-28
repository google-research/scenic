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

"""Standalone eval."""

from collections.abc import Sequence
import time
from typing import Optional

from absl import flags
from absl import logging
from clu import metric_writers
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.lang4video import main_lib
from scenic.projects.lang4video import util

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'eval_while_training', False,
    'Whether this evaluation is run alongside a training job. If False, assume '
    'this is running standalone, so we only only run a single eval step on the '
    'last checkpoint.')
flags.DEFINE_integer(
    'job_index', None,
    'The number of this eval job. `None` means to evaluate all configs under'
    ' "evaluation_configs", -1 means to run the evaluate of the dataset used'
    ' for training but on the validation split, otherwise it indicates the'
    ' index of the evaluation config to run.')

flags.DEFINE_bool('use_a_different_writer', False,
                  'Whether to use a different metric writer for this file.')
flags.DEFINE_string('eval_suffix', '', 'Suffix for the metric writer.')

RATE_LIMIT = 10  # In seconds.


def evaluate_config(
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
    evaluation_config: Optional[ml_collections.ConfigDict] = None,
) -> None:
  """Calls `main_lib.main` with a mix of `config` and `evaluation_config`."""
  with (config_to_use := util.safe_deepcopy_config(config)).unlocked():
    if evaluation_config:
      util.update_config_including_refs(config_to_use, evaluation_config)

      prefix_suffix = evaluation_config.get('dataset_canonical_name',
                                            evaluation_config.dataset_name)
      # TODO(sacastro): change 'valid' for 'test' if using the test set?
      config_to_use.writer_prefix = f'valid/{prefix_suffix}'
    else:  # It means we evaluate the dataset we already have in `config`.
      config_to_use.trainer_name = 'zero_shot_text_to_visual_retrieval_trainer'

    config_to_use.dataset_configs.load_train = False
    config_to_use.dataset_configs.load_val = True
    config_to_use.dataset_configs.load_test = False

    config_to_use.dataset_configs.cache_processed = True

    if config_to_use.model.get('text_encoder', {}).get('pretraining_mode'):
      # TODO(sacastro): this is super ad-hoc.
      config_to_use.model.text_encoder.compute_mlm = False

    if 'lr_configs' in config_to_use:
      # If present, this key may trigger getting the training steps, which
      # there's none for evaluation. So we remove it.
      del config_to_use['lr_configs']

    # Dicts inside lists/tuples can't be frozen. Plus, this is unused:
    del config_to_use['evaluation_configs']

  main_lib.main(rng=None, config=config_to_use, workdir=workdir, writer=writer)


def eval_main(
    rng: jnp.ndarray,  # pylint: disable=unused-argument
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for evals."""
  del rng

  assert jax.process_count() == 1, 'Eval only supports a single host.'

  if FLAGS.eval_while_training:
    # This eval job is being run in parallel to a training job, so we want to
    # poll for new checkpoints until the training job is done.
    # TODO(sacastro)
    pass
  else:
    # This eval job is run separately from XManager on a previously trained
    # model, so we want to evaluate just the last checkpoint.
    pass

  if FLAGS.use_a_different_writer:
    # Use a different metrics writer than the one Scenic provides so that we
    # don't corrupt the original train directory. This one will write to a
    # subdirectory.
    dir_suffix = 'xid-wid'

    workdir_w_suffix = workdir.rstrip('/') + FLAGS.eval_suffix
    writer = metric_writers.create_default_writer(
        workdir_w_suffix, collection=f'eval-{dir_suffix}', asynchronous=True)

  did_initial_run = False
  last_evaluated_checkpoint = None

  train_batch_size = None

  while True:
    latest_checkpoint = checkpoints.latest_checkpoint(workdir)

    if FLAGS.eval_while_training:
      if FLAGS.job_index == -1 and not train_batch_size and (
          train_batch_size := ''):
        config.retrieval_batch_size = train_batch_size
        logging.info('The training batch size is %d', train_batch_size)

      # Check if we need to evaluate, wait, or stop:
      # Decide is_final_checkpoint, or handle errors or wait for new ckpt.
      if training_failed := False:  # pylint: disable=unused-variable
        # Stop evaluator if parent training job has failed:
        logging.warning('Training has failed. Stopping eval job.')
        # TODO(sacastro): stop
        return
      elif FLAGS.job_index == -1 and not train_batch_size:
        logging.info('Waiting to get the training batch size…')
        time.sleep(RATE_LIMIT)
        continue
      elif latest_checkpoint != last_evaluated_checkpoint:
        # Checkpoint is new --> proceed to evaluation:
        logging.info('Found new checkpoint: %s', latest_checkpoint)
      elif last_evaluated_checkpoint is None and not did_initial_run:
        logging.info('Found no checkpoint. '
                     'Evaluating with the initial model to warm the cache up.')
        did_initial_run = True
      elif (final_checkpoint := ''
           ) is not None and latest_checkpoint == final_checkpoint:
        break
      elif final_checkpoint == 'None':
        raise ValueError(
            'Training job reported final checkpoint "None", meaning that no '
            'checkpoint was saved. Check if config.checkpoint == True.')
      else:  # No new checkpoint.
        # Checkpoint is old, but not final --> wait, then re-start loop:
        # No need to check final checkpoint in this case
        logging.info('Waiting for a new checkpoint…')
        time.sleep(RATE_LIMIT)
        continue

    if FLAGS.job_index is None:
      evaluation_configs = config.evaluation_configs
    elif FLAGS.job_index == -1:
      evaluation_configs = [None]
    else:
      evaluation_config = config.evaluation_configs[FLAGS.job_index]
      if isinstance(evaluation_config, Sequence):
        evaluation_configs = evaluation_config
      else:
        evaluation_configs = [evaluation_config]

    for evaluation_config in evaluation_configs:
      evaluate_config(
          config=config,
          evaluation_config=evaluation_config,
          workdir=workdir,
          writer=writer)

    last_evaluated_checkpoint = latest_checkpoint

    if not FLAGS.eval_while_training:
      break

  logging.info('Exiting.')


if __name__ == '__main__':
  app.run(main=eval_main)
