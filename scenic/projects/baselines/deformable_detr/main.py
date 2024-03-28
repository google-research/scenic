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

"""Main file for DeformableDETR."""

from typing import Any

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.baselines.deformable_detr import input_pipeline_detection  # pylint: disable=unused-import
from scenic.projects.baselines.deformable_detr import trainer
from scenic.projects.baselines.deformable_detr.model import DeformableDETRModel
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS

_TRAIN = flags.DEFINE_bool('train', True, 'Run training or just eval.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'deformable_detr':
    return DeformableDETRModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def resolve(obj):
  """Resolve `FieldReference`s in `obj`.

  We do not have a good way to resolve every `FieldReference` in `obj`. The
  function will raise a TypeError if it encounters something it cannot handle.

  Args:
    obj: The object to resolve.

  Returns:
    The resolved object.
  """
  if obj is None:
    return None
  elif isinstance(obj, (int, float, str, bool)):
    return obj
  elif isinstance(obj, ml_collections.FieldReference):
    return resolve(obj.get())
  elif isinstance(obj, ml_collections.ConfigDict):
    resolved = ml_collections.ConfigDict()
    for key, value in obj.items():
      resolved[key] = resolve(value)
    return resolved
  elif isinstance(obj, list):
    return [resolve(x) for x in obj]
  elif isinstance(obj, tuple):
    return tuple([resolve(x) for x in obj])
  else:
    raise TypeError(f'Cannot resolve type {type(obj)}')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the DETR project."""
  if _BATCH_SIZE.value is not None:
    config.batch_size = _BATCH_SIZE.value
  config = resolve(config)
  config = ml_collections.FrozenConfigDict(config)

  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  if _TRAIN.value:
    trainer.train_and_evaluate(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)
  else:
    trainer.evaluate(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)


if __name__ == '__main__':
  app.run(main=main)
