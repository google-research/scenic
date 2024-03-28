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

"""For log_train_summary."""

from typing import Any, Callable, Dict, Tuple, Sequence, Optional, Mapping, Union

from clu import metric_writers
import jax
import jax.numpy as jnp

from scenic.train_lib import train_utils

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]
PRNGKey = jnp.ndarray


def log_train_summary(step: int,
                      *,
                      writer: metric_writers.MetricWriter,
                      train_metrics: Sequence[Dict[str, Tuple[float, int]]],
                      train_images: Any = None,
                      extra_training_logs: Optional[Sequence[Dict[str,
                                                                  Any]]] = None,
                      metrics_normalizer_fn: Optional[
                          Callable[[Dict[str, Tuple[float, int]], str],
                                   Dict[str, float]]] = None,
                      prefix: str = 'train',
                      step_idx: Optional[int] = None,
                      key_separator: str = '_') -> Dict[str, float]:
  """Computes and logs train metrics."""
  if step_idx is None:
    step_idx = step

  def fmt(i, p):
    return f'%.{p}d' % i

  if train_images is not None:
    train_images = train_utils.stack_forest(
        train_images)  # key -> list(ndarray)
    train_images = jax.tree_util.tree_map(lambda x: jnp.concatenate(x)[:4],
                                          train_images)
    new_train_images = {}
    for key, value in train_images.items():
      for (batch_idx, image) in enumerate(value):
        new_train_images[
            f'{key}/bi{fmt(batch_idx,p=2)}/s{fmt(step_idx,p=8)}'] = image[0,
                                                                          ...]

    writer.write_images(step, new_train_images)

  ##### Prepare metrics:
  # Get metrics from devices:
  train_metrics = train_utils.stack_forest(train_metrics)
  # Compute the sum over all examples in all batches:
  train_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(),
                                                 train_metrics)
  # Normalize metrics by the total number of exampels:
  metrics_normalizer_fn = metrics_normalizer_fn or train_utils.normalize_metrics_summary
  train_metrics_summary = metrics_normalizer_fn(train_metrics_summary, 'train')

  ##### Prepare additional training logs:
  # If None, set to an empty dictionary.
  extra_training_logs = extra_training_logs or {}
  train_logs = train_utils.stack_forest(extra_training_logs)

  # Metrics:
  writer.write_scalars(
      step, {
          key_separator.join((prefix, key)): val
          for key, val in train_metrics_summary.items()
      })
  # Additional logs:
  writer.write_scalars(step,
                       {key: val.mean() for key, val in train_logs.items()})

  writer.flush()
  return train_metrics_summary
