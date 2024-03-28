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

"""Main function for the Scenic project Lang4Video."""

import getpass
import os
from typing import Optional

from absl import flags
from absl import logging
import chex
from clu import metric_writers
from flax.training import checkpoints
import jax
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
import ml_collections

from scenic.common_lib import debug_utils
from scenic.model_lib import models
from scenic.projects.lang4video import util
from scenic.projects.lang4video.model.image_text_model import ImageTextModel
import scenic.projects.lang4video.opt  # pylint: disable=unused-import
import scenic.projects.lang4video.pp_ops  # pylint: disable=unused-import
from scenic.projects.lang4video.trainer import visual_text_trainer
from scenic.projects.lang4video.trainer import visual_text_with_text_pretraining_trainer
from scenic.projects.lang4video.trainer import zero_shot_classification_trainer
from scenic.projects.lang4video.trainer import zero_shot_text_to_visual_retrieval_trainer
from scenic.projects.lang4video.trainer.train_utils import get_cached_fn
from scenic.train_lib import train_utils
from scenic.train_lib import trainers


FLAGS = flags.FLAGS
PLATFORM = flags.DEFINE_string(
    'platform_type',
    None,
    'Platform type.',
)
MEGACORE = flags.DEFINE_bool('megacore', True, 'Megacore.')

FINAL_CKPT_ARTIFACT_DESCRIPTION = 'Final checkpoint'

models.ALL_MODELS['image_text'] = ImageTextModel

trainers.ALL_TRAINERS['visual_text_trainer'] = visual_text_trainer.train
trainers.ALL_TRAINERS['visual_text_with_text_pretraining_trainer'] = (
    visual_text_with_text_pretraining_trainer.train
)
trainers.ALL_TRAINERS['zero_shot_classification_trainer'] = (
    zero_shot_classification_trainer.evaluate
)
trainers.ALL_TRAINERS['zero_shot_text_to_visual_retrieval_trainer'] = (
    zero_shot_text_to_visual_retrieval_trainer.evaluate
)




# Max size 1 in case there's a miss, to not run OOM.
_get_dataset_cached = get_cached_fn(train_utils.get_dataset, max_size=1)


def main(rng: Optional[jnp.ndarray], config: ml_collections.ConfigDict,
         workdir: str, writer: metric_writers.MetricWriter) -> None:
  """Main function for the Scenic."""
  with chex.fake_pmap_and_jit(
      enable_pmap_patching=not config.get('enable_pmap_and_jit'),
      enable_jit_patching=not config.get('enable_pmap_and_jit')):
    with config.unlocked():
      config.device_memory_gbs = util.get_device_memory(
          platform_type=FLAGS.platform_type, megacore=FLAGS.megacore)
      config.device_count = jax.device_count()

    if 'lr_configs' in config:
      # TODO(sacastro): there must be a better way to check if this is the
      #   train job. Maybe some flag?
      logging.info(
          'Logging the training batch size (%d),'
          ' so the eval jobs can know about it.', config.batch_size)
      writer.write_scalars(0, {'batch_size': config.batch_size})

    if config.get('debug_train'):
      logging.warning('DEBUG MODE IS ENABLED!')
      debug_utils.enable_jax_debugging_flags()

    if (config.get('use_jax_compilation_cache', True) and
        hasattr(jax.devices()[0].client, 'runtime_type')):
      jax_cache_dir = os.path.join(workdir, 'jax_cache', 'ttl=30d')
      logging.info('JAX compilation cache path: %s', jax_cache_dir)
      compilation_cache.set_cache_dir(jax_cache_dir)

    model_cls = models.get_model_cls(config.model_name)
    assert model_cls is ImageTextModel

    if rng is None:
      data_rng = None
    else:
      data_rng, rng = jax.random.split(rng)

    if config.checkpoint and data_rng is not None:
      # When restoring from a checkpoint, change the dataset seed to ensure
      # that the example order is new. With deterministic data, this ensures
      # enough randomization and in the future with deterministic data +
      # random access, we can feed the global step to the dataset loader to
      # always continue reading the rest of the data if we resume a job that
      # was interrupted.
      checkpoint_path = checkpoints.latest_checkpoint(workdir)
      logging.info('CHECKPOINT PATH: %s', checkpoint_path)
      if checkpoint_path is not None:
        global_step = train_utils.checkpoint_path_step(checkpoint_path) or 0
        logging.info('Folding global_step %s into dataset seed.', global_step)
        data_rng = jax.random.fold_in(data_rng, global_step)

    # We set some defaults for the dataset loading:

    if (dataset_configs := config.get('dataset_configs')) is None:
      with config.unlocked():
        config.dataset_configs = dataset_configs = ml_collections.ConfigDict()


    # Resolve the field references. This is necessary because
    # `ref_util.FieldReference` comparison operations return
    # `ml_collections.FieldReference` objects, which in turn don't support
    # `__bool__`, and make any comparison fail. This behavior makes cached
    # functions fail.
    #
    # It creates an extra copy, but it's not an issue.
    #
    # Also, this needs to be done before freezing because otherwise it doesn't
    # copy the field `_configdict` from `ml_collections.FrozenConfigDict`.
    hashable_config = config.copy_and_resolve_references()

    hashable_config = ml_collections.FrozenConfigDict(hashable_config)

    dataset = _get_dataset_cached(
        config=hashable_config,
        data_rng=data_rng,
        dataset_service_address=FLAGS.dataset_service_address)

    if lr_configs := config.get('lr_configs'):
      total_steps = train_utils.get_num_training_steps(
          config, dataset.meta_data)[0]

      with lr_configs.unlocked():
        if ((warmup_steps := lr_configs.get('warmup_steps')) and
            isinstance(warmup_steps, float) and 0 <= warmup_steps <= 1):
          lr_configs.warmup_steps = warmup_steps * total_steps

        if ((steps_per_cycle := lr_configs.get('steps_per_cycle')) and
            isinstance(steps_per_cycle, float) and 0 <= steps_per_cycle <= 1):
          lr_configs.steps_per_cycle = steps_per_cycle * total_steps

    trainers.get_trainer(config.trainer_name)(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)
