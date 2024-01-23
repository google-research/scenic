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

"""Tests for masked layers."""

import dataclasses
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import masked_layers as masked


def _pad_norm_assert_shape(outputs):
  """Test that the spatial shape output is correct for masked BN/GN."""
  outputs, spatial_shape = outputs
  # Does it have the correct shape?
  np.testing.assert_equal(
      spatial_shape.shape,
      MaskedLayersTest.INPUTS_SHAPE_SMALL.shape)
  # Inferred spatial shape has the correct values.
  np.testing.assert_allclose(
      spatial_shape,
      MaskedLayersTest.INPUTS_SHAPE_SMALL,
      atol=0)

  _, h, w, _ = MaskedLayersTest.INPUTS_SMALL.shape
  norm_unpad = outputs[:, :h, :w, :]      # Region without padding.
  norm_right_pad = outputs[:, :h, w:, :]  # Right padded region.
  norm_bottom_pad = outputs[:, h:, :w]    # Bottom padded region.
  return norm_unpad, [norm_right_pad, norm_bottom_pad]


def _pad_norm_assert_noshape(outputs):
  """Assert Masked BN/GN w/o spatial shape returns None shape."""
  outputs, spatial_shape = outputs
  assert spatial_shape is None
  return outputs, []


def _pad_norm_noshape(outputs):
  """Equivalent of `_pad_norm_assert_noshape` for normal BN/GN."""
  return outputs, []


@dataclasses.dataclass
class NormSpec:
  """Used for consicely parameterizing Batch/Group Norm tests (see test_norm).

  Attributes:
    cls: BatchNorm/GroupNorm class to create.
    ctor_kwargs: Class constructor kwargs.
    init_kwargs: Initializer (i.e. `cls.init_with_output`) kwargs.
    process_fn: Output processing function. Takes output of Batch/Group Norm
      (normalized outputs and a shape tensor in case masked layers; or just
      normalized outputs), optionally asserts that spatial shapes are correct,
      and returns unpadded (masked removed) outputs and a list of padded parts.
  """
  cls: Union[Type[nn.BatchNorm], Type[masked.BatchNorm],
             Type[nn.GroupNorm], Type[masked.GroupNorm]]
  ctor_kwargs: Dict[str, Any]
  init_kwargs: Dict[str, Any]
  process_fn: Callable[[jnp.ndarray], Tuple[jnp.ndarray, Sequence[jnp.ndarray]]]


class MaskedLayersTest(parameterized.TestCase):
  """Tests for modules in masked_layers.py."""

  SMALL_SIZE, LARGE_SIZE, PADDED_SIZE = 16, 27, 35

  INPUTS_SHAPE_SMALL = 8, SMALL_SIZE, SMALL_SIZE, 16
  INPUTS_SHAPE_LARGE = 8, LARGE_SIZE, LARGE_SIZE, 16

  INPUTS_SMALL = np.random.normal(size=INPUTS_SHAPE_SMALL)
  INPUTS_LARGE = np.random.normal(size=INPUTS_SHAPE_LARGE)

  INPUTS_SMALL_PADDED = np.pad(
      INPUTS_SMALL,
      [(0, 0), (0, PADDED_SIZE - SMALL_SIZE), (0, PADDED_SIZE - SMALL_SIZE),
       (0, 0)],
      'constant')
  INPUTS_LARGE_PADDED = np.pad(
      INPUTS_LARGE,
      [(0, 0), (0, PADDED_SIZE - LARGE_SIZE), (0, PADDED_SIZE - LARGE_SIZE),
       (0, 0)],
      'constant')
  INPUTS_PADDED = np.concatenate(
      [INPUTS_SMALL_PADDED, INPUTS_LARGE_PADDED],
      axis=0)

  INPUTS_SHAPE_SMALL = np.array([[SMALL_SIZE, SMALL_SIZE]] * 8)
  INPUTS_SHAPE_LARGE = np.array([[LARGE_SIZE, LARGE_SIZE]] * 8)
  SPATIAL_SHAPE = np.concatenate(
      [INPUTS_SHAPE_SMALL, INPUTS_SHAPE_LARGE], axis=0)

  POOL_FN_DICT = {'avg': (nn.avg_pool, masked.avg_pool),
                  'max': (nn.max_pool, masked.max_pool)}

  @parameterized.named_parameters([
      ('same_pad_11_11', (1, 1), (1, 1), 'SAME', None, None),
      ('same_pad_11_22', (1, 1), (2, 2), 'SAME', None, None),
      ('valid_pad_33_13', (3, 3), (1, 3), 'VALID', None, None),
      ('valid_pad_53_21', (5, 3), (2, 1), 'VALID', None, None),
      ('valid_pad_33_13_kd12', (3, 3), (1, 3), 'VALID', None, (1, 2)),
      ('valid_pad_53_21_kd33', (5, 3), (2, 1), 'VALID', None, (3, 3)),
      ('num_pad_33_13', (3, 3), (1, 3), [(4, 7), (1, 1)], None, None),
      ('num_pad_53_21', (5, 3), (2, 1), [(0, 0), (1, 1)], None, None),
      ('num_pad_33_13_id21', (3, 3), (1, 3), [(4, 7), (1, 1)], (2, 1), None),
      ('num_pad_53_21_id33', (5, 3), (2, 1), [(0, 0), (1, 1)], (3, 3), None),
      ('num_pad_33_13_id21_kd12', (3, 3), (1, 3), [(4, 7), (1, 1)], (2, 1),
       (1, 2)),
      ('num_pad_53_21_id33_kd25', (5, 3), (2, 1), [(0, 0), (1, 1)], (3, 3),
       (2, 5)),
  ])
  def test_unpadded_conv_eq_masked_padded(self, kernel_size, strides, padding,
                                          input_dilation, kernel_dilation):
    """Conv on unpadded data and conv on padded and masked data are same."""
    conv_args = {
        'features': 64,
        'kernel_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'use_bias': True,
        'input_dilation': input_dilation,
        'kernel_dilation': kernel_dilation,
        'kernel_init': nn.initializers.ones,
        'bias_init': nn.initializers.ones,
    }

    rng = random.PRNGKey(0)
    conv, masked_conv = nn.Conv(**conv_args), masked.Conv(**conv_args)

    # It is OK to re-init since we're keeping the rng constant.
    output_small, conv_params = conv.init_with_output(rng, self.INPUTS_SMALL)
    output_large = conv.apply(conv_params, self.INPUTS_LARGE)

    (outputs_padded, spatial_shape), _ = masked_conv.init_with_output(
        rng,
        self.INPUTS_PADDED,
        spatial_shape=self.SPATIAL_SHAPE)

    # Inferred spatial shape has the right shape.
    self.assertEqual(
        spatial_shape.shape,
        (self.INPUTS_PADDED.shape[0], 2))

    # Inferred spatial shape has the right values.
    n_small, n_large = output_small.shape[0], output_large.shape[0]
    np.testing.assert_allclose(
        spatial_shape[:n_small, ...],
        np.stack([np.array(output_small.shape[1:-1])] * n_small, axis=0),
        atol=0)
    np.testing.assert_allclose(
        spatial_shape[-n_large:, ...],
        np.stack([np.array(output_large.shape[1:-1])] * n_large, axis=0),
        atol=0)

    # Masked output has the right values in the *un*masked region.
    ind_small = [slice(s) for s in output_small.shape[1:-1]]
    ind_large = [slice(s) for s in output_large.shape[1:-1]]
    ind_small = tuple([slice(n_small)] + ind_small + [slice(None)])
    ind_large = tuple([slice(n_small, None)] + ind_large + [slice(None)])

    np.testing.assert_allclose(
        output_small, outputs_padded[ind_small], atol=1e-5)
    np.testing.assert_allclose(
        output_large, outputs_padded[ind_large], atol=1e-5)

    # Masked output has the right values in the masked region.
    ind_small = [slice(s, None) for s in output_small.shape[1:-1]]
    ind_large = [slice(s, None) for s in output_large.shape[1:-1]]
    ind_small = tuple([slice(n_small)] + ind_small + [slice(None)])
    ind_large = tuple([slice(n_small, None)] + ind_large + [slice(None)])

    np.testing.assert_allclose(jnp.zeros_like(outputs_padded[ind_small]),
                               outputs_padded[ind_small],
                               atol=1e-5)
    np.testing.assert_allclose(jnp.zeros_like(outputs_padded[ind_large]),
                               outputs_padded[ind_large],
                               atol=1e-5)

  @parameterized.named_parameters([
      ('same_pad_11_11', (1, 1), (1, 1), 'SAME', None, None),
      ('same_pad_11_22', (1, 1), (2, 2), 'SAME', None, None),
      ('same_pad_33_23', (3, 3), (2, 3), 'SAME', None, None),
      ('same_pad_53_32', (5, 3), (3, 2), 'SAME', None, None),
      ('same_pad_33_23_kd12', (3, 3), (2, 3), 'SAME', None, (1, 2)),
      ('same_pad_53_32_kd43', (5, 3), (3, 2), 'SAME', None, (4, 3)),
      ('valid_pad_33_13', (3, 3), (1, 3), 'VALID', None, None),
      ('valid_pad_53_21', (5, 3), (2, 1), 'VALID', None, None),
      ('valid_pad_33_13_kd12', (3, 3), (1, 3), 'VALID', None, (1, 2)),
      ('valid_pad_53_21_kd33', (5, 3), (2, 1), 'VALID', None, (3, 3)),
      ('num_pad_33_13', (3, 3), (1, 3), [(4, 7), (1, 1)], None, None),
      ('num_pad_53_21', (5, 3), (2, 1), [(0, 0), (1, 1)], None, None),
      ('num_pad_33_13_id21', (3, 3), (1, 3), [(4, 7), (1, 1)], (2, 1), None),
      ('num_pad_53_21_id33', (5, 3), (2, 1), [(0, 0), (1, 1)], (3, 3), None),
      ('num_pad_33_13_id21_kd12', (3, 3), (1, 3), [(4, 7), (1, 1)], (2, 1),
       (1, 2)),
      ('num_pad_53_21_id33_kd25', (5, 3), (2, 1), [(0, 0), (1, 1)], (3, 3),
       (2, 5)),
  ])
  def test_masked_conv_without_spatial_shape(
      self, kernel_size, strides, padding, input_dilation, kernel_dilation):
    """Masked conv without spatial shape behaves same as normal conv."""
    conv_args = {
        'features': 64,
        'kernel_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'use_bias': True,
        'input_dilation': input_dilation,
        'kernel_dilation': kernel_dilation,
        'kernel_init': nn.initializers.ones,
        'bias_init': nn.initializers.ones,
    }

    rng = random.PRNGKey(0)
    conv, masked_conv = nn.Conv(**conv_args), masked.Conv(**conv_args)

    output_conv, _ = conv.init_with_output(rng, self.INPUTS_SMALL)
    (output_masked_conv, _), _ = masked_conv.init_with_output(
        rng, self.INPUTS_SMALL)
    np.testing.assert_allclose(output_conv, output_masked_conv, atol=1e-5)

  def test_masked_same_conv_raises(self):
    """Masked convolutions with 'SAME' padding are not supported."""
    conv_args = {
        'features': 64,
        'kernel_size': (3, 3),
        'strides': (3, 3),
        'padding': 'SAME',
        'use_bias': True,
        'input_dilation': None,
        'kernel_dilation': None,
        'kernel_init': nn.initializers.ones,
        'bias_init': nn.initializers.ones,
    }

    rng = random.PRNGKey(0)
    masked_conv = masked.Conv(**conv_args)

    with self.assertRaises(NotImplementedError):
      masked_conv.init(
          rng,
          self.INPUTS_PADDED,
          spatial_shape=self.SPATIAL_SHAPE)

  @parameterized.named_parameters([
      ('same_pad_11_11_avg', 'avg', (1, 1), (1, 1), 'SAME'),
      ('same_pad_11_22_avg', 'avg', (1, 1), (2, 2), 'SAME'),
      ('valid_pad_33_13_avg', 'avg', (3, 3), (1, 3), 'VALID'),
      ('valid_pad_53_21_avg', 'avg', (5, 3), (2, 1), 'VALID'),
      ('num_pad_33_13_avg', 'avg', (3, 3), (1, 3), [(4, 7), (1, 1)]),
      ('num_pad_53_21_avg', 'avg', (5, 3), (2, 1), [(0, 0), (1, 1)]),
      ('same_pad_11_11_max', 'max', (1, 1), (1, 1), 'SAME'),
      ('same_pad_11_22_max', 'max', (1, 1), (2, 2), 'SAME'),
      ('valid_pad_33_13_max', 'max', (3, 3), (1, 3), 'VALID'),
      ('valid_pad_53_21_max', 'max', (5, 3), (2, 1), 'VALID'),
      ('num_pad_33_13_max', 'max', (3, 3), (1, 3), [(4, 7), (1, 1)]),
      ('num_pad_53_21_max', 'max', (5, 3), (2, 1), [(0, 0), (1, 1)]),
  ])
  def test_unpadded_pool_eq_masked_padded(
      self, pool_fn, window_shape, strides, padding):
    """Pool on unpadded data and pool on padded and masked data are same."""
    pool_fn, masked_pool_fn = self.POOL_FN_DICT[pool_fn]
    output_small = pool_fn(
        self.INPUTS_SMALL, window_shape, strides, padding=padding)
    output_large = pool_fn(
        self.INPUTS_LARGE, window_shape, strides, padding=padding)

    outputs_padded, spatial_shape = masked_pool_fn(
        self.INPUTS_PADDED,
        window_shape,
        strides,
        padding=padding,
        spatial_shape=self.SPATIAL_SHAPE)

    # Inferred spatial shape has the right shape.
    self.assertEqual(
        spatial_shape.shape,
        (self.INPUTS_PADDED.shape[0], 2))

    # Inferred spatial shape has the right values.
    n_small, n_large = output_small.shape[0], output_large.shape[0]
    np.testing.assert_allclose(
        spatial_shape[:n_small, ...],
        np.stack([np.array(output_small.shape[1:-1])] * n_small, axis=0),
        atol=0)
    np.testing.assert_allclose(
        spatial_shape[-n_large:, ...],
        np.stack([np.array(output_large.shape[1:-1])] * n_large, axis=0),
        atol=0)

    # Masked output has the right values in the *un*masked region.
    ind_small = [slice(s) for s in output_small.shape[1:-1]]
    ind_large = [slice(s) for s in output_large.shape[1:-1]]
    ind_small = tuple([slice(n_small)] + ind_small + [slice(None)])
    ind_large = tuple([slice(n_small, None)] + ind_large + [slice(None)])

    np.testing.assert_allclose(
        output_small, outputs_padded[ind_small], atol=1e-5)
    np.testing.assert_allclose(
        output_large, outputs_padded[ind_large], atol=1e-5)

    # Masked output has the right values in the masked region.
    ind_small = [slice(s, None) for s in output_small.shape[1:-1]]
    ind_large = [slice(s, None) for s in output_large.shape[1:-1]]
    ind_small = tuple([slice(n_small)] + ind_small + [slice(None)])
    ind_large = tuple([slice(n_small, None)] + ind_large + [slice(None)])

    np.testing.assert_allclose(jnp.zeros_like(outputs_padded[ind_small]),
                               outputs_padded[ind_small],
                               atol=1e-5)
    np.testing.assert_allclose(jnp.zeros_like(outputs_padded[ind_large]),
                               outputs_padded[ind_large],
                               atol=1e-5)

  @parameterized.named_parameters([
      ('same_pad_12_11_avg', 'avg', (1, 2), (1, 1), 'SAME'),
      ('same_pad_21_22_max', 'max', (2, 1), (2, 2), 'SAME'),
  ])
  def test_masked_same_pool_raises(
      self, pool_fn, window_shape, strides, padding):
    """Masked pool with 'SAME' padding is not supported."""
    _, masked_pool_fn = self.POOL_FN_DICT[pool_fn]

    with self.assertRaises(NotImplementedError):
      masked_pool_fn(
          self.INPUTS_PADDED, window_shape, strides,
          padding=padding, spatial_shape=self.SPATIAL_SHAPE)

  @parameterized.named_parameters([
      # Batch Norm tests.
      ('masked_bn_shape_eq_bn',
       NormSpec(cls=nn.BatchNorm,
                ctor_kwargs={
                    'use_running_average': False,
                    'use_bias': True,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL},
                process_fn=_pad_norm_noshape),
       NormSpec(cls=masked.BatchNorm,
                ctor_kwargs={
                    'use_running_average': False,
                    'use_bias': True,
                    'use_scale': True,
                    'spatial_norm': True},
                init_kwargs={
                    'x': INPUTS_SMALL_PADDED,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape)),
      ('masked_bn_shape_eq_masked_bn_noshape',
       NormSpec(cls=masked.BatchNorm,
                ctor_kwargs={
                    'use_running_average': False,
                    'use_bias': True,
                    'use_scale': True,
                    'spatial_norm': True},
                init_kwargs={
                    'x': INPUTS_SMALL},
                process_fn=_pad_norm_assert_noshape),
       NormSpec(cls=masked.BatchNorm,
                ctor_kwargs={
                    'use_running_average': False,
                    'use_bias': True,
                    'use_scale': True,
                    'spatial_norm': True},
                init_kwargs={
                    'x': INPUTS_SMALL_PADDED,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape)),
      ('masked_bn_shape_eq_masked_bn_shape_nospatial',
       NormSpec(cls=masked.BatchNorm,
                ctor_kwargs={
                    'use_running_average': False,
                    'use_bias': True,
                    'use_scale': True,
                    'spatial_norm': True},
                init_kwargs={
                    'x': INPUTS_SMALL,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape),
       NormSpec(cls=masked.BatchNorm,
                ctor_kwargs={
                    'use_running_average': False,
                    'use_bias': True,
                    'use_scale': True,
                    'spatial_norm': False},
                init_kwargs={
                    'x': INPUTS_SMALL,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape)),

      # Group Norm tests.
      ('masked_gn_shape_eq_gn',
       NormSpec(cls=nn.GroupNorm,
                ctor_kwargs={
                    'num_groups': 8,
                    'use_bias': True,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL},
                process_fn=_pad_norm_noshape),
       NormSpec(cls=masked.GroupNorm,
                ctor_kwargs={
                    'num_groups': 8,
                    'use_bias': True,
                    'spatial_norm': True,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL_PADDED,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape)),
      ('masked_gn_shape_eq_masked_gn_noshape',
       NormSpec(cls=masked.GroupNorm,
                ctor_kwargs={
                    'num_groups': 8,
                    'use_bias': True,
                    'spatial_norm': True,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL},
                process_fn=_pad_norm_assert_noshape),
       NormSpec(cls=masked.GroupNorm,
                ctor_kwargs={
                    'num_groups': 8,
                    'use_bias': True,
                    'spatial_norm': True,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL_PADDED,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape)),
      ('masked_gn_shape_eq_masked_gn_shape_nosptial',
       NormSpec(cls=masked.GroupNorm,
                ctor_kwargs={
                    'num_groups': 8,
                    'use_bias': True,
                    'spatial_norm': False,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape),
       NormSpec(cls=masked.GroupNorm,
                ctor_kwargs={
                    'num_groups': 8,
                    'use_bias': True,
                    'spatial_norm': True,
                    'use_scale': True},
                init_kwargs={
                    'x': INPUTS_SMALL_PADDED,
                    'spatial_shape': INPUTS_SHAPE_SMALL},
                process_fn=_pad_norm_assert_shape)),
  ])
  def test_norm(self, norm1_spec, norm2_spec):
    """Test Batch/Group Norm unaffected by padding."""
    norm1 = norm1_spec.cls(**norm1_spec.ctor_kwargs)
    norm2 = norm2_spec.cls(**norm2_spec.ctor_kwargs)

    # It is OK to re-init since we're keeping the rng constant.
    rng = random.PRNGKey(0)
    norm1_outputs, norm1_params = norm1.init_with_output(
        rng, **norm1_spec.init_kwargs)
    norm2_outputs, norm2_params = norm2.init_with_output(
        rng, **norm2_spec.init_kwargs)

    # Inferred spatial shape has the right shape.
    norm1_unpad, norm1_pad = norm1_spec.process_fn(norm1_outputs)
    norm2_unpad, norm2_pad = norm2_spec.process_fn(norm2_outputs)

    # Unpadded parts of both outputs are the same.
    np.testing.assert_allclose(norm1_unpad, norm2_unpad, atol=1e-5)

    # All padded parts are zero.
    for part in norm1_pad + norm2_pad:
      np.testing.assert_allclose(part, np.zeros_like(part), atol=1e-5)

    # Run a second time and repeat all the checks. This is necessary for BN
    # because it is stateful; and does not change the output for GN.
    norm1_outputs, norm1_params = norm1.apply(
        norm1_params, mutable=['batch_stats'], **norm1_spec.init_kwargs)
    norm2_outputs, norm2_params = norm2.apply(
        norm2_params, mutable=['batch_stats'], **norm2_spec.init_kwargs)

    norm1_unpad, norm1_pad = norm1_spec.process_fn(norm1_outputs)
    norm2_unpad, norm2_pad = norm2_spec.process_fn(norm2_outputs)

    # Unpadded parts of both outputs are the same.
    np.testing.assert_allclose(norm1_unpad, norm2_unpad, atol=1e-5)

    # All padded parts are zero.
    for part in norm1_pad + norm2_pad:
      np.testing.assert_allclose(part, np.zeros_like(part), atol=1e-5)


if __name__ == '__main__':
  absltest.main()
