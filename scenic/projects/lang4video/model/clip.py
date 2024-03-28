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

"""CLIP image and text encoders."""

from typing import Optional

import flax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import jax
import jax.numpy as jnp

from scenic.projects.baselines.clip import layers as clip_layers
from scenic.projects.baselines.clip import model as clip_model
from scenic.projects.lang4video.model.base_encoders import ImageEncoder
from scenic.projects.lang4video.model.base_encoders import TextEncoder

# We can't simply load a CLIP model and re-use an existing submodule. If we do
# so, it would load extra parameters (in this case just `logit_scale` because
# `setup` methods are going to be called recursively but not compact methods as
# they are lazy; this one is the only non-lazy one). If we have extra args, when
# we load the pretrained params, we would have different parameters than
# expected, this is error-prone. Better to have exactly what we use. Plus, there
# aren't extra params in memory.
#
# Note we can't do it neither in a `setup` method, because the constructed CLIP
# object will be unbound, thus the submodules undefined.

clip_model.CONFIGS['debug'] = {
    'embed_dim': 16,
    'vocab_size': 49408,
    'vision_num_layers': 1,
    'vision_features': 128,
    'vision_patch_size': 112,
    'text_features': 16,
    'text_num_heads': 2,
    'text_num_layers': 1,
}


class ClipImageEncoder(ImageEncoder):
  """CLIP image encoder."""

  config_name: str
  normalize: bool = True
  return_all_tokens: bool = False

  @nn.compact
  def __call__(
      self,
      image: jnp.ndarray,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    config = clip_model.CONFIGS[self.config_name]

    # We leave the default eps because the BatchNorm and LayerNorm are always
    # computed in FP32.
    if isinstance(config['vision_num_layers'], (tuple, list)):
      clip_image_encoder = clip_layers.ModifiedResNet(
          num_layers=config['vision_num_layers'],
          features=config['vision_features'],
          num_heads=config['vision_features'] * 32 // 6,
          out_features=(None if config.get('vision_return_map') else
                        config['embed_dim']),
          # dtype=self.dtype,
          name='visual')
    else:
      clip_image_encoder = clip_layers.VisionTransformer(
          patch_size=config['vision_patch_size'],
          features=config['vision_features'],
          num_layers=config['vision_num_layers'],
          num_heads=config['vision_features'] // 64,
          out_features=(None if config.get('vision_return_map') else
                        config['embed_dim']),
          # dtype=self.dtype,
          name='visual')

    x, feature_map = clip_image_encoder(image)

    if self.return_all_tokens:
      # We leave the default eps because the LayerNorm is always computed in
      # FP32.
      return nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(feature_map)
    else:
      if self.normalize:
        eps = jnp.finfo(x.dtype).eps
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
      else:
        return x

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    if self.config_name == 'debug':
      return flax.core.freeze({}), flax.core.freeze({})
    else:
      clip_model_vars = clip_model.load_model_vars(self.config_name)

      model_vars = {
          k: {k2: v2 for k2, v2 in v.items() if k2 == 'visual'
             } for k, v in clip_model_vars.items()
      }

      params = model_vars.pop('params')
      params = jax.tree_util.tree_map(lambda x: x.astype(self.dtype), params)
      params = flax.core.freeze(params)

      model_state = flax.core.freeze(model_vars)

      return params, model_state


class ClipTextEncoder(TextEncoder):
  """CLIP text encoder."""

  config_name: str
  normalize: bool = True

  @nn.compact
  def __call__(
      self,
      text: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    del mask  # CLIP model computes its own mask.
    config = clip_model.CONFIGS[self.config_name]
    # We leave the default eps because the LayerNorm is always computed in FP32.
    clip_text_encoder = clip_layers.TextEncoder(
        out_features=config['embed_dim'],
        vocab_size=config['vocab_size'],
        features=config['text_features'],
        num_layers=config['text_num_layers'],
        num_heads=config['text_num_heads'],
        # dtype=self.dtype,
        name='text')
    x = clip_text_encoder(text)
    if self.normalize:
      eps = jnp.finfo(x.dtype).eps
      return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
    else:
      return x

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    if self.config_name == 'debug':
      return flax.core.freeze({}), flax.core.freeze({})
    else:
      clip_model_vars = clip_model.load_model_vars(self.config_name)

      model_vars = {
          k: {k2: v2 for k2, v2 in v.items() if k2 == 'text'
             } for k, v in clip_model_vars.items()
      }

      params = model_vars.pop('params')
      params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
      params = flax.core.freeze(params)

      model_state = flax.core.freeze(model_vars)

      return params, model_state


# TODO(sacastro): load the temperature from the pretrained model.
