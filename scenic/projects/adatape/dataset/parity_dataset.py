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

"""Data generators for the parity task."""

import functools
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets


def generate_parity_sample(batch_size, seq_len):
  """Generate one sample for parity task.

  Args:
    batch_size: Determines the batch size.
    seq_len: Determines the sequence length of parity vector.

  Yields:
    One Sample.

  """
  rng = jax.random.PRNGKey(0)
  while True:
    rng, _ = jax.random.split(rng)
    # Parity:
    sample = jax.random.choice(
        rng,
        a=jnp.array((1.0, 0.0, -1.0), jnp.float32),
        shape=(batch_size, seq_len))
    label = jnp.sum(jnp.equal(sample, 1.0), axis=-1).astype(jnp.int32) % 2
    sample = jax.nn.one_hot(sample+1, 3).astype(jnp.float32)
    yield {'inputs': sample, 'label': label}


def generate_parity_eval_sample(batch_size, seq_len):
  """Generate one sample for parity task.

  Args:
    batch_size: Determines the batch size.
    seq_len: Determines the sequence length of parity vector.

  Yields:
    One Sample.

  """
  rng = jax.random.PRNGKey(42)
  while True:
    rng, _ = jax.random.split(rng)
    # Parity:
    sample = jax.random.choice(
        rng,
        a=jnp.array((1.0, 0.0, -1.0), jnp.float32),
        shape=(batch_size, seq_len))
    label = jnp.sum(jnp.equal(sample, 1.0), axis=-1).astype(jnp.int32) % 2
    sample = jax.nn.one_hot(sample+1, 3).astype(jnp.float32)
    yield {'inputs': sample, 'label': label}


@datasets.add_dataset('parity')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=None,
                rng=None,
                dataset_configs=None,
                dataset_service_address=None):
  """Returns generators for the PARITY train and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: We will not use it.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del shuffle_seed
  del rng
  del dataset_service_address
  # Init configs.
  if dataset_configs and dataset_configs.get('seq_len'):
    seq_len = dataset_configs['seq_len']
  else:
    seq_len = 32
  if dataset_configs and dataset_configs.get('num_train_examples'):
    num_train_examples = dataset_configs['num_train_examples']
  else:
    num_train_examples = 64000
  num_eval_examples = num_train_examples//10

  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  train_iter = generate_parity_sample(batch_size=batch_size, seq_len=seq_len)
  train_iter = map(shard_batches, train_iter)
  eval_iter = generate_parity_eval_sample(
      batch_size=eval_batch_size, seq_len=seq_len)
  eval_iter = map(shard_batches, eval_iter)

  # Parity:
  input_shape = (-1, seq_len, 3)

  meta_data = {
      'num_classes': 2,
      'input_shape': input_shape,
      'num_train_examples': num_train_examples,
      'num_eval_examples': num_eval_examples,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': False,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)

