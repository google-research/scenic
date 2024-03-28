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

r"""Main script for the Layout Denoise project."""

from absl import flags
from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.layout_denoise import model
from scenic.projects.layout_denoise import trainer
from scenic.projects.layout_denoise.datasets import dataset

FLAGS = flags.FLAGS


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the LayoutDenoise project."""
  model_cls = model.LayoutModel

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  rng = jax.random.PRNGKey(config.rng_seed)
  logging.info('rng: %s', rng)
  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  eval_batch_size = config.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the number of devices ({device_count})')

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  dataset_dict = {}
  for name, cfg in config.datasets.items():
    data_rng, rng = jax.random.split(rng)
    ds = dataset.get_dataset(
        batch_size=local_batch_size,
        eval_batch_size=eval_local_batch_size,
        num_shards=jax.local_device_count(),
        dtype_str=cfg.data_dtype_str,
        rng=data_rng,
        config=config,
        dataset_configs=cfg)
    dataset_dict[name] = ds

  trainer.train(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset_dict=dataset_dict,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
