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

"""Tests for model.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
from scenic.projects.token_learner import model


class TokenLearnerTest(parameterized.TestCase):
  """Tests for modules in token-learner model.py."""

  @parameterized.named_parameters(
      ('32_tokens', 32),
      ('111_tokens', 111),
  )
  def test_dynamic_tokenizer(self, num_tokens):
    """Tests TokenLearner module."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 224, 224, 64))
    tokenizer = functools.partial(model.TokenLearnerModule,
                                  num_tokens=num_tokens)
    tokenizer_vars = tokenizer().init(rng, x)
    y = tokenizer().apply(tokenizer_vars, x)
    # Test outputs shape.
    self.assertEqual(y.shape, (x.shape[0], num_tokens, x.shape[-1]))

  @parameterized.named_parameters(
      ('encoder_image', (2, 16, 192), 'dynamic', 1, 8, model.EncoderMod),
      ('encoder_video_temporal_dims_1',
       (2, 16, 192), 'video', 1, 8, model.EncoderMod),
      ('encoder_video_temporal_dims_2',
       (2, 32, 192), 'video', 2, 8, model.EncoderMod),
      ('encoder_video_temporal_dims_4',
       (2, 64, 192), 'video', 4, 8, model.EncoderMod),
      ('encoder_fusion_image',
       (2, 16, 192), 'dynamic', 1, 8, model.EncoderModFuser),
      ('encoder_fusion_video_temporal_dims_1',
       (2, 16, 192), 'video', 1, 8, model.EncoderModFuser),
      ('encoder_fusion_video_temporal_dims_2',
       (2, 32, 192), 'video', 2, 8, model.EncoderModFuser),
      ('encoder_fusion_video_temporal_dims_4',
       (2, 64, 192), 'video', 4, 8, model.EncoderModFuser),
  )
  def test_encoder(self, input_shape, tokenizer_type,
                   temporal_dimensions, num_tokens, encoder_function):
    """Tests shapes of TokenLearner Encoder (with and without TokenFuser)."""
    rng = random.PRNGKey(0)
    dummy_input = jnp.ones(input_shape)
    encoder = functools.partial(
        encoder_function,
        num_layers=3,
        mlp_dim=192,
        num_heads=3,
        tokenizer_type=tokenizer_type,
        temporal_dimensions=temporal_dimensions,
        num_tokens=num_tokens,
        tokenlearner_loc=2)
    encoder_vars = encoder().init(rng, dummy_input)
    y = encoder().apply(encoder_vars, dummy_input)

    if encoder_function == model.EncoderMod:
      if tokenizer_type == 'dynamic':
        expected_shape = (input_shape[0], num_tokens, input_shape[2])
      elif tokenizer_type == 'video':
        expected_shape = (
            input_shape[0], num_tokens * temporal_dimensions, input_shape[2])
      else:
        raise ValueError('Unknown tokenizer type.')
    elif encoder_function == model.EncoderModFuser:
      expected_shape = input_shape
    else:
      raise ValueError('Unknown encoder function.')

    self.assertEqual(y.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
