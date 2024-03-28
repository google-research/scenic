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

"""Tests for model_utils.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.projects.fast_vit import model_utils


class AttentionLayersTest(parameterized.TestCase):
  """Tests for modules in model_utils.py."""

  def test_linformer_encoder_self_attention(self):
    """Tests EncoderSelfAttention."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 16, 32))
    n_heads = 2
    encoder_self_attention_def = functools.partial(
        model_utils.LinformerEncoderAttention, num_heads=n_heads)
    encoder_vars = encoder_self_attention_def().init(rng, x, deterministic=True)
    y = encoder_self_attention_def().apply(encoder_vars, x, deterministic=True)
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  def test_linformer_encoder_self_attention_w_dropout(self):
    """Tests EncoderSelfAttention with dropout."""
    rng = random.PRNGKey(0)
    rng, dropout_rng = random.split(rng)
    x = jnp.ones((4, 16, 32))
    n_heads = 2
    encoder_self_attention_def = functools.partial(
        model_utils.LinformerEncoderAttention,
        num_heads=n_heads,
        dropout_rate=0.1)
    encoder_vars = encoder_self_attention_def().init(rng, x, deterministic=True)
    y = encoder_self_attention_def().apply(
        encoder_vars, x, deterministic=False, rngs={'dropout': dropout_rng})
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters([
      ('test_softmax', 'softmax'),
      ('test_generalized', 'generalized'),
  ])
  def test_performer_encoder_self_attention(self, attention_fn_cls):
    """Tests PerformerEncoderAttention."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 16, 32))
    n_heads = 2
    encoder_self_attention_def = functools.partial(
        model_utils.PerformerEncoderAttention,
        num_heads=n_heads,
        attention_fn_cls=attention_fn_cls)
    encoder_vars = encoder_self_attention_def().init(
        rng, x, x, deterministic=True)
    y = encoder_self_attention_def().apply(
        encoder_vars, x, x, deterministic=True)
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters([
      ('test_softmax', 'softmax'),
      ('test_generalized', 'generalized'),
  ])
  def test_performer_encoder_self_attention_w_dropout(self, attention_fn_cls):
    """Tests PerformerEncoderAttention with dropout."""
    rng = random.PRNGKey(0)
    rng, dropout_rng = random.split(rng)
    x = jnp.ones((4, 16, 32))
    n_heads = 2
    encoder_self_attention_def = functools.partial(
        model_utils.PerformerEncoderAttention,
        num_heads=n_heads,
        attention_fn_cls=attention_fn_cls)
    encoder_vars = encoder_self_attention_def().init(
        rng, x, x, deterministic=True)
    y = encoder_self_attention_def().apply(
        encoder_vars, x, x, deterministic=False, rngs={'dropout': dropout_rng})
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters([('test_axi1', 1), ('test_axi2', 2)])
  def test_axial_reshaping_utils(self, axis):
    """Tests fo get_axial_1d_input and get_axial_2d_input."""
    input_shape = (4, 8, 16, 32)  # Shape = `[bs, h, w, c]`
    inputs_2d = jnp.array(np.random.normal(size=input_shape))
    inputs_1d = model_utils.get_axial_1d_input(inputs_2d, axis=axis)
    inputs_back_to_2d = model_utils.get_axial_2d_input(
        inputs_1d, axis=axis, two_d_shape=input_shape)
    self.assertTrue(jnp.array_equal(inputs_2d, inputs_back_to_2d))


class TopKTokenSelectorTest(parameterized.TestCase):
  """Tests for token selector module."""

  @parameterized.named_parameters([
      ('pool_unselected', True, False, False, 6),
      ('only_selected', False, False, False, 5),
      ('pool_unselected_exclude_cls', True, True, False, 7),
      ('only_selected_exclude_cls', False, True, False, 6),
      ('pool_unselected_sample', True, False, True, 6),
      ('only_selected_sample', False, False, True, 5),
      ('pool_unselected_exclude_cls_sample', True, True, True, 7),
      ('only_selected_exclude_cls_sample', False, True, True, 6),
  ])
  def test_top_k_selector(self,
                          pool_unselected_tokens,
                          exclude_cls,
                          sample_tokens,
                          expected_output_len):
    """Tests Top-K selector."""
    rng, sample_rng = random.split(random.PRNGKey(0))
    x = jnp.ones((4, 16, 32))
    top_k = 5
    top_selector = functools.partial(
        model_utils.TopKTokenSelector,
        top_k=top_k,
        sample_tokens=sample_tokens,
        pool_unselected_tokens=pool_unselected_tokens,
        exclude_cls=exclude_cls,
    )
    variable = top_selector().init(rng, x, train=False)
    y = top_selector().apply(
        variable, x, train=True, rngs={'dropout': sample_rng})
    # Test outputs shape.
    expected_shape = (4, expected_output_len, 32)
    self.assertEqual(y.shape, expected_shape)

  @parameterized.named_parameters([
      ('replacement',
       (32, 6, 10), 7, True, (32, 6, 7), False, False, False),
      ('replacement_nonunique',
       (32, 6, 10), 11, None, (32, 6, 11), False, False, True),
      ('no_replacement',
       (32, 6, 10), 10, False, (32, 6, 10), False, True, False),
      ('no_replacement_raises',
       (32, 6, 10), 11, False, None, True, None, None),
  ])
  def test_sample_categorial(self,
                             logit_shape,
                             num_samples,
                             replacement,
                             expected_shape,
                             expected_raise,
                             expected_unique,
                             expeted_nonunique):
    """Test categorial sampler."""
    rng, sample_rng = random.split(random.PRNGKey(0))
    logits = random.normal(rng, logit_shape)

    kwargs = {}
    if replacement is not None:
      kwargs['replacement'] = replacement

    if expected_raise:
      with self.assertRaises(ValueError):
        samples = model_utils.sample_categorical(
            sample_rng, logits, num_samples, **kwargs)
    else:
      samples = model_utils.sample_categorical(
          sample_rng, logits, num_samples, **kwargs)
      self.assertEqual(samples.shape, expected_shape)

      if expected_unique or expeted_nonunique:
        samples = jnp.reshape(samples, (-1, expected_shape[-1]))
        samples = jax.device_get(samples)
        for sample in samples:
          if expected_unique:
            self.assertEqual(len(set(sample.tolist())), len(sample))
          if expeted_nonunique:
            self.assertLess(len(set(sample.tolist())), len(sample))


if __name__ == '__main__':
  absltest.main()
