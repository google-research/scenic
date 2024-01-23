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

"""Tests for utilities used in individual datasets and in dataset_utils.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils


class DatasetUtilsTest(parameterized.TestCase):
  """Tests Dataset Utilities."""

  @parameterized.named_parameters(
      # ('pixel_level_pre_pad_mask', True, (28, 28), 28 * 28, True),
      # ('pixel_levefel', True, (28, 28), 28 * 28, False),
      ('example_level', False, (), 1, False),)
  def test_maybe_pad_batch(self, pixel_level, batch_mask_shape, mask_sum_coef,
                           pre_padding_mask):
    """Tests maybe_pad_batch."""
    desired_bs = 32
    partial_bs = 13

    def make_fake_batches():
      # Make dummy batches
      complete_batch = {
          'inputs': jnp.array(np.random.normal(size=(desired_bs, 28, 28, 3))),
          'label': jnp.array(np.random.normal(size=(desired_bs, 10)))
      }
      partial_batch = {
          'inputs': jnp.array(np.random.normal(size=(partial_bs, 28, 28, 3))),
          'label': jnp.array(np.random.normal(size=(partial_bs, 10)))
      }

      complete_batch_mask, partial_batch_mask = None, None
      if pre_padding_mask:
        complete_batch_mask = jnp.broadcast_to(
            jnp.eye(28, 28), (desired_bs, 28, 28))
        complete_batch['batch_mask'] = complete_batch_mask
        partial_batch_mask = jnp.broadcast_to(
            jnp.eye(28, 28), (partial_bs, 28, 28))
        partial_batch['batch_mask'] = partial_batch_mask

      return (complete_batch, partial_batch, complete_batch_mask,
              partial_batch_mask)

    ######## Test complete training batches
    (complete_batch, partial_batch, complete_batch_mask,
     partial_batch_mask) = make_fake_batches()
    outputs = dataset_utils.maybe_pad_batch(
        complete_batch,
        train=True,
        batch_size=desired_bs,
        pixel_level=pixel_level)
    # Check output shape:
    self.assertEqual(outputs['inputs'].shape, (desired_bs, 28, 28, 3))
    # Check batch_mask:
    self.assertEqual(outputs['batch_mask'].shape,
                     (desired_bs,) + batch_mask_shape)
    if pre_padding_mask:
      self.assertEqual(outputs['batch_mask'].sum(), complete_batch_mask.sum())
    else:
      self.assertEqual(outputs['batch_mask'].sum(),
                       float(desired_bs * mask_sum_coef))

    print(complete_batch.keys())
    ######## Test partial training batches
    # Assert that the code throws an error as we dont handle partial training
    # batches in the codebase:
    with self.assertRaises(ValueError):
      _ = dataset_utils.maybe_pad_batch(
          partial_batch,
          train=True,
          batch_size=desired_bs,
          pixel_level=pixel_level)

    ######## Test complete test batches
    (complete_batch, partial_batch, complete_batch_mask,
     partial_batch_mask) = make_fake_batches()
    outputs = dataset_utils.maybe_pad_batch(
        complete_batch,
        train=False,
        batch_size=desired_bs,
        pixel_level=pixel_level)
    # Check output shape:
    self.assertEqual(outputs['inputs'].shape, (desired_bs, 28, 28, 3))
    # Check batch_mask:
    self.assertEqual(outputs['batch_mask'].shape,
                     (desired_bs,) + batch_mask_shape)
    if pre_padding_mask:
      self.assertEqual(outputs['batch_mask'].sum(), complete_batch_mask.sum())
    else:
      self.assertEqual(outputs['batch_mask'].sum(),
                       float(desired_bs * mask_sum_coef))

    ######## Test partial test batches
    outputs = dataset_utils.maybe_pad_batch(
        partial_batch,
        train=False,
        batch_size=desired_bs,
        pixel_level=pixel_level)
    # Check output shape:
    self.assertEqual(outputs['inputs'].shape, (desired_bs, 28, 28, 3))

    # check output padding
    expected_out_pad = jnp.array(np.zeros((desired_bs - partial_bs, 28, 28, 3)))
    out_pad = outputs['inputs'][partial_bs:, :, :, :]
    self.assertTrue(jnp.array_equal(out_pad, expected_out_pad))

    # Check batch_mask:
    self.assertEqual(outputs['batch_mask'].shape,
                     (desired_bs,) + batch_mask_shape)

    batch_mask = jnp.concatenate([
        jnp.array(np.ones((partial_bs,) + batch_mask_shape)),
        jnp.array(np.zeros((desired_bs - partial_bs,) + batch_mask_shape))
    ],
                                 axis=0)
    if pre_padding_mask:
      padded_pre_padding_mask = jnp.concatenate([
          partial_batch_mask,
          jnp.array(np.zeros((desired_bs - partial_bs,) + batch_mask_shape))
      ],
                                                axis=0)
      batch_mask *= padded_pre_padding_mask

    self.assertTrue(jnp.array_equal(outputs['batch_mask'], batch_mask))

  @parameterized.named_parameters(
      ('NHWC-jnp', 'NHWC', (16, 4, 5, 32), True),
      ('NTHWC-jnp', 'NTHWC', (16, 2, 4, 5, 32), True),
      ('NHWC-np', 'NHWC', (16, 4, 5, 32), False),
      ('NTHWC-np', 'NTHWC', (16, 2, 4, 5, 32), False),
  )
  def test_mixup(self, image_format, inputs_shape, jax_numpy):
    """Tests mixup augmentation for different input formats and numpys."""
    bs = inputs_shape[0]
    num_classes = 10

    if jax_numpy:
      np_backend = jnp
      mixup_fn = jax.jit(
          functools.partial(
              dataset_utils.mixup,
              alpha=1.0,
              image_format=image_format,
              rng=jax.random.PRNGKey(0)))
    else:
      np_backend = np
      mixup_fn = functools.partial(
          dataset_utils.mixup, alpha=1.0, image_format=image_format, rng=None)

    # Make a fake batch:
    inputs = np_backend.array(
        np.concatenate((np.zeros(shape=(bs // 2,) + inputs_shape[1:]),
                        np.ones(shape=(bs // 2,) + inputs_shape[1:])),
                       axis=0))
    labels = np_backend.array(
        jax.nn.one_hot(
            np.concatenate(
                (
                    np.ones(shape=(bs // 2,)),  # class 1
                    np.ones(shape=(bs // 2,)) * 2  # class 2
                ),
                axis=0),
            num_classes))
    fake_batch = {'inputs': inputs, 'label': labels}

    # Apply mixup:
    mixedup_batch = mixup_fn(fake_batch)

    self.assertEqual(mixedup_batch['inputs'].shape, inputs_shape)
    self.assertEqual(mixedup_batch['label'].shape, (bs, num_classes))


if __name__ == '__main__':
  absltest.main()
