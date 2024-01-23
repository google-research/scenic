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

"""Tests for attention_layers.py."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import attention_layers


class AttentionLayersTest(parameterized.TestCase):
  """Tests for modules in attention_layers.py."""

  @parameterized.named_parameters([
      ('test_same_qk', (10, 28, 4, 32), (10, 28, 4, 32)),
      ('test_different_qk', (10, 12, 4, 32), (10, 13, 4, 32)),
  ])
  def test_dot_product_attention(self, q_shape, k_shape):
    """Test dot_product_attention function."""
    rng = random.PRNGKey(0)
    v_shape = k_shape[:-1] + (64,)
    expected_output_shape = q_shape[:-1] + (v_shape[-1],)

    query = jnp.array(np.random.normal(size=q_shape))
    key = jnp.array(np.random.normal(size=k_shape))
    value = jnp.array(np.random.normal(size=v_shape))
    y = attention_layers.dot_product_attention(
        query,
        key,
        value,
        deterministic=False,
        dropout_rng=rng,
        capture_attention_weights=False)
    # Test outputs shape.
    self.assertEqual(y.shape, expected_output_shape)

  def test_multihead_attention(self):
    """Tests MultiHeadAttention."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 16, 32))
    n_heads = 2
    layer = attention_layers.MultiHeadAttention(num_heads=n_heads)
    variables = layer.init(rng, x, x, deterministic=True)
    y = layer.apply(variables, x, x, deterministic=True)
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  def test_multihead_attention_hidden_size_not_divisible_by_heads(self):
    """Tests MultiHeadAttention."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 16, 30))
    n_heads = 4
    layer = attention_layers.MultiHeadAttention(
        num_heads=n_heads, enforce_hidden_size_divisible_by_heads=False)
    variables = layer.init(rng, x, x, deterministic=True)
    self.assertTupleEqual(variables['params']['query']['kernel'].shape,
                          (30, 4, 7))
    self.assertTupleEqual(variables['params']['key']['kernel'].shape,
                          (30, 4, 7))
    self.assertTupleEqual(variables['params']['value']['kernel'].shape,
                          (30, 4, 7))
    self.assertTupleEqual(variables['params']['out']['kernel'].shape,
                          (4, 7, 30))
    y = layer.apply(variables, x, x, deterministic=True)
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  def test_multihead_attention_w_dropout(self):
    """Tests MultiHeadAttention with dropout."""
    rng = random.PRNGKey(0)
    rng, dropout_rng = random.split(rng)
    x = jnp.ones((4, 16, 32))
    n_heads = 2
    layer = attention_layers.MultiHeadAttention(
        num_heads=n_heads, dropout_rate=0.1)
    variables = layer.init(rng, x, x, deterministic=True)
    y = layer.apply(
        variables, x, x, deterministic=False, rngs={'dropout': dropout_rng})
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters([
      ('test_learned', nn.initializers.ones),
      ('test_sinusoidal', None),
  ])
  def test_add_1d_positional_embedding(self, posemb_init):
    """Tests Add1DPositionEmbedding."""
    rng = random.PRNGKey(0)
    input_shape = (4, 16, 32)
    inputs = jnp.array(np.random.normal(size=input_shape))

    # Test output after adding positional embedding.
    layer = (attention_layers.Add1DPositionEmbedding(posemb_init=posemb_init))
    variables = layer.init(rng, inputs)
    outputs = layer.apply(variables, inputs)

    # Test output shape.
    self.assertEqual(outputs.shape, input_shape)

    if posemb_init is not None:
      # Test added learned embeddings.
      # Note that we initialize them with nn.initializers.ones.
      expected_added_pos_emb = jnp.ones(input_shape, dtype=inputs.dtype)
      added_pos_emb = outputs - inputs
      np.testing.assert_allclose(
          added_pos_emb, expected_added_pos_emb, atol=1e-6)

      # Test embeddings shape
      self.assertEqual(variables['params']['pos_embedding'].shape,
                       (1,) + input_shape[1:])

  def test_add_2d_positional_embedding(self):
    """Tests Add2DPositionEmbedding."""
    rng = random.PRNGKey(0)
    input_shape = (4, 8, 16, 32)
    inputs = jnp.ones(input_shape)

    # Test output after adding positional embedding.
    layer = attention_layers.Add2DPositionEmbedding(
        posemb_init=nn.initializers.ones)
    variables = layer.init(rng, inputs)
    outputs = layer.apply(variables, inputs)

    # Test output shape.
    self.assertEqual(outputs.shape, input_shape)

    # Test added embeddings.
    # Note that we initialize them with nn.initializers.ones.
    expected_added_pos_emb = jnp.ones(input_shape, dtype=inputs.dtype)
    added_pos_emb = outputs - inputs
    np.testing.assert_allclose(added_pos_emb, expected_added_pos_emb, atol=1e-6)

    # Test embeddings shape.
    self.assertEqual(variables['params']['row_pos_embedding'].shape,
                     (input_shape[2], input_shape[-1] // 2))
    self.assertEqual(variables['params']['col_pos_embedding'].shape,
                     (input_shape[1], input_shape[-1] // 2))

  @parameterized.named_parameters([
      ('test_2d', (10, 28, 32, 4, 32), (10, 28, 32, 4, 32)),
      ('test_3d', (10, 12, 28, 32, 9, 32), (10, 12, 28, 32, 9, 32)),
  ])
  def test_axial_dot_product_attention_has_expected_shape(
      self, q_shape, k_shape):
    """Test axial_dot_product_attention function."""
    v_shape = k_shape[:-1] + (64,)
    expected_output_shape = q_shape[:-1] + (v_shape[-1],)

    query = jnp.array(np.random.normal(size=q_shape))
    key = jnp.array(np.random.normal(size=k_shape))
    value = jnp.array(np.random.normal(size=v_shape))
    y = attention_layers.axial_dot_product_attention(
        query, key, value, deterministic=True)
    # Test outputs shape:
    self.assertEqual(y.shape, expected_output_shape)

  @parameterized.named_parameters([
      ('test_1d', (7,)),
      ('test_2d', (3, 7)),
      ('test_3d', (3, 5, 7)),
  ])
  def test_relative_attention_bias(self, nd_shape):
    """Test axial_dot_product_attention function."""
    num_heads = 2
    bias_layer = attention_layers.RelativeAttentionBias(
        num_heads=num_heads, nd_shape=nd_shape,
        initializer=nn.initializers.normal())
    rng = random.PRNGKey(0)
    variables = bias_layer.init(rng)
    bias = bias_layer.apply(variables)

    length = np.prod(nd_shape)
    self.assertEqual((num_heads, length, length), bias.shape)

    bias_nd = bias.reshape((num_heads,) + nd_shape + nd_shape)
    for i in range(len(nd_shape)):
      bias_crop = bias_nd
      for _ in range(i + 1, len(nd_shape)):
        # Crop until last dim is dim to be checked.
        bias_crop = bias_crop[:, :, ..., 0]
      for k in range(nd_shape[i] - 1):
        np.testing.assert_array_equal(bias_crop[:, k, ..., :-1],
                                      bias_crop[:, k + 1, ..., 1:])
      bias_nd = bias_nd[:, 0]

    # Now plug this bias into multi-head attention.
    layer = attention_layers.MultiHeadAttention(num_heads=num_heads)
    input_shape = (4, length, num_heads * 2)
    inputs = jnp.array(np.random.normal(size=input_shape))
    variables = layer.init(rng, inputs, inputs)
    layer.apply(variables, inputs, inputs, attention_bias=bias)


if __name__ == '__main__':
  absltest.main()
