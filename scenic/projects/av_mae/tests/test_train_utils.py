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

"""Unit tests for train_utils.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
from scenic.projects.av_mae import train_utils


class TestTrainUtils(parameterized.TestCase):
  """Tests for modules in train_utils.py."""

  @parameterized.named_parameters(
      ('num_columns_2', 2),
      ('num_columns_4', 4),
  )
  def test_generate_image_grid(self, n_columns):
    """Tests that shapes of generate image grid are correct."""

    rng = random.PRNGKey(0)
    height = 16
    width = 16
    patch_size = (4, 4)
    num_tokens = int(height / patch_size[0] * width / patch_size[1])
    batch = 8
    channels = patch_size[0] * patch_size[1] * 3

    input_shape = (batch, num_tokens, channels)
    inputs = random.uniform(rng, shape=input_shape, minval=-1, maxval=1)
    input_mask = random.bernoulli(rng, p=0.2, shape=(batch, num_tokens))

    output = train_utils.generate_image_grid(
        target=inputs,
        prediction=inputs,
        prediction_masks=input_mask,
        patch_size=patch_size,
        n_columns=n_columns,
        input_size=(height, width, 3))

    expected_shape = (batch / n_columns * 6 * height, n_columns * width, 3)
    self.assertEqual(output.shape, expected_shape)
    self.assertEqual(output.dtype, jnp.uint8)
    self.assertTrue(jnp.all(jnp.greater_equal(output, 0)))
    self.assertTrue(jnp.all(jnp.less_equal(output, 255)))

  @parameterized.named_parameters(
      ('NHWC', (16, 4, 5, 32)),
      ('NTHWC', (16, 2, 4, 5, 32))
  )
  def test_mixup(self, inputs_shape):
    """Tests syntax errors and shape for mixup and cutmix."""
    bs = inputs_shape[0]
    num_classes = 10

    mixup_fn = jax.jit(
        functools.partial(
            train_utils.mixup_cutmix,
            mixup_alpha=1.0,
            cutmix_alpha=1.0,
            rng=jax.random.PRNGKey(0),
            switch_prob=0.5,
            label_smoothing=0.2))

    # Make a fake batch.
    inputs = jnp.concatenate((jnp.zeros(shape=(bs // 2,) + inputs_shape[1:]),
                              jnp.ones(shape=(bs // 2,) + inputs_shape[1:])),
                             axis=0)
    labels = jax.nn.one_hot(
        jnp.concatenate(
            (
                jnp.ones(shape=(bs // 2,)),  # class 1
                jnp.ones(shape=(bs // 2,)) * 2  # class 2
            ),
            axis=0),
        num_classes)
    fake_batch = {'inputs': inputs, 'label': labels}

    # Apply mixup.
    mixedup_batch = mixup_fn(fake_batch)

    self.assertEqual(mixedup_batch['inputs'].shape, inputs_shape)
    self.assertEqual(mixedup_batch['label'].shape, (bs, num_classes))


if __name__ == '__main__':
  absltest.main()
