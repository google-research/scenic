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

"""T5 text encoder."""

import dataclasses
from typing import Optional

import flax
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.lang4video.model.base_encoders import TextEncoder
from t5x import checkpoints as t5x_checkpoints

from flaxformer.architectures.t5 import t5_1_1


# Add the checkpoints:
CHECKPOINT = {}

CONFIGS = {
    'debug':
        t5_1_1.Config(
            embedding_dim=16,
            mlp_dim=2,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ),
    'small':
        t5_1_1.SMALL_CONFIG,
    'base':
        t5_1_1.BASE_CONFIG,
    'large':
        t5_1_1.LARGE_CONFIG,
    'xl':
        t5_1_1.XL_CONFIG,
    'xxl':
        t5_1_1.XXL_CONFIG,
    'lm100k-small':
        t5_1_1.SMALL_CONFIG,
    'lm100k-base':
        t5_1_1.BASE_CONFIG,
    'lm100k-large':
        t5_1_1.LARGE_CONFIG,
    'lm100k-xl':
        t5_1_1.XL_CONFIG,
    'lm100k-xxl':
        t5_1_1.XXL_CONFIG,
}


class T5TextEncoder(TextEncoder):
  """T5 text encoder."""

  config_name: str
  dtype: jnp.dtype = jnp.bfloat16
  return_all_tokens: bool = False

  @nn.compact
  def __call__(
      self,
      text: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    # We leave the default eps because the LayerNorm is always computed in FP32.
    t5 = t5_1_1.encoder_decoder(
        **dataclasses.asdict(CONFIGS[self.config_name]), dtype=self.dtype)
    # TODO(sacastro): normalize?

    # T5 computes its own mask. So we don't pass it.
    encoded_tokens = t5.encode(text, enable_dropout=train)

    if self.return_all_tokens:
      return encoded_tokens
    else:
      if mask is None:
        mask = text > 0

      # We average the token embeddings. In Sentence-T5
      # (https://arxiv.org/abs/2108.08877), they show the mean works great.
      num_tokens = mask.sum(axis=-1, keepdims=True)
      num_tokens = jnp.maximum(num_tokens, jnp.ones_like(num_tokens))
      return (encoded_tokens * mask[..., jnp.newaxis]).sum(axis=-2) / num_tokens

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    if self.config_name == 'debug':
      return flax.core.freeze({}), flax.core.freeze({})
    else:
      t5_checkpoint_state = t5x_checkpoints.load_t5x_checkpoint(
          CHECKPOINTS[self.config_name])
      # TODO(sacastro): use `t5_checkpoint_state['state']['param_states']`?
      #   For initializing the optimizer stats.
      params = flax.core.freeze({
          'EncoderDecoder_0': {
              k: v
              for k, v in t5_checkpoint_state['target'].items()
              if k in {'encoder', 'token_embedder'}
          }
      })
      params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
      return params, flax.core.freeze({})
