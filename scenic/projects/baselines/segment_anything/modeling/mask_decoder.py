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

r"""Sam mask decoder.

Pytorch reference:

https://github.com/facebookresearch/segment-anything/blob/HEAD/\
segment_anything/modeling/mask_decoder.py

"""

import flax.linen as nn
import jax.numpy as jnp
from scenic.projects.baselines.segment_anything.modeling import transformer


class MaskDecoder(nn.Module):
  """Sam mask decoder."""

  transformer_dim: int = 256
  num_multimask_outputs: int = 3
  iou_head_depth: int = 3
  iou_head_hidden_dim: int = 256

  def setup(self):
    self.iou_token = self.param(
        'iou_token.weight',
        nn.initializers.normal(stddev=1.),
        (1, self.transformer_dim))
    self.mask_tokens = self.param(
        'mask_tokens.weight',
        nn.initializers.normal(stddev=1.),
        (self.num_multimask_outputs + 1, self.transformer_dim))
    self.output_upscaling = OutputScaling(
        transformer_dim=self.transformer_dim, name='output_upscaling')

    self.output_hypernework_mlps = [
        MLP(hidden_dim=self.iou_head_hidden_dim,
            output_dim=self.transformer_dim // 8, num_layers=3,
            name=f'output_hypernetworks_mlps.{i}',
           ) for i in range(self.num_multimask_outputs + 1)]

    self.iou_prediction_head = MLP(
        hidden_dim=self.iou_head_hidden_dim,
        output_dim=self.num_multimask_outputs + 1,
        num_layers=self.iou_head_depth,
        name='iou_prediction_head')

    self.transformer = transformer.TwoWayTransformer(name='transformer')

  def predict_masks(
      self, image_embeddings, image_pe,
      sparse_prompt_embeddings, dense_prompt_embeddings):
    """Predict masks for a single image.

    Args:
      image_embeddings: (H, W, embed_dim)
      image_pe: (H, W, embed_dim)
      sparse_prompt_embeddings: (num_prompts, num_points, embed_dim)
      dense_prompt_embeddings: (num_prompts, H, W, embed_dim)
    Returns:
      masks: (num_prompts, num_multimask_outputs + 1, h', w')
      iou_pred: (num_prompts, num_multimask_outputs + 1)
    """
    output_tokens = jnp.concatenate(
        [self.iou_token, self.mask_tokens],
        axis=0)  # (num_multimask_outputs + 2, transformer_dim)
    num_prompts = sparse_prompt_embeddings.shape[0]
    output_tokens = jnp.broadcast_to(
        output_tokens[None],
        (num_prompts, self.num_multimask_outputs + 2, self.transformer_dim))
    tokens = jnp.concatenate(
        [output_tokens, sparse_prompt_embeddings], axis=1,
    )  # (num_prompts, num_multimask_outputs + 2 + num_points, embed_dim)

    src = jnp.repeat(
        image_embeddings[None], tokens.shape[0],
        axis=0)  # (num_prompts, H, W, D)
    src = src + dense_prompt_embeddings
    pos_src = jnp.repeat(
        image_pe[None], tokens.shape[0], axis=0)  # (num_prompts, H, W, D)
    num_prompts, h, w, d = src.shape

    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1: (1 + self.num_multimask_outputs + 1), :]

    src = src.reshape(num_prompts, h, w, d)
    upscaled_embedding = self.output_upscaling(src)  # (num_prompts, h', w', d)
    hyper_in_list = []
    for i in range(self.num_multimask_outputs + 1):
      hyper_in_list.append(
          self.output_hypernework_mlps[i](
              mask_tokens_out[:, i, :])  # (num_prompts, d)
      )
    hyper_in = jnp.stack(hyper_in_list, axis=1)  # (num_prompts, num_masks, d)
    num_prompts, h, w, d = upscaled_embedding.shape
    masks = hyper_in @ upscaled_embedding.reshape(
        num_prompts, h * w, d).transpose(
            0, 2, 1)  # (num_prompts, num_masks, h'w')
    masks = masks.reshape(num_prompts, self.num_multimask_outputs + 1, h, w)

    iou_pred = self.iou_prediction_head(iou_token_out)
    return masks, iou_pred

  @nn.compact
  def __call__(
      self, image_embeddings, image_pe,
      sparse_prompt_embeddings, dense_prompt_embeddings,
      multimask_output: bool = True):
    """Forward model for a single image.

    Args:
      image_embeddings: (H, W, 3)
      image_pe: (H, W, D)
      sparse_prompt_embeddings: (num_prompts, num_points, embed_dim)
      dense_prompt_embeddings: (num_prompts, H, W, embed_dim)
      multimask_output: bool
    Returns:
      masks: (num_prompts, num_multimask_outputs, h', w'),
        num_multimask_outputs = 3 if multimask_output is True, otherwise 1.
      iou_pred: (num_prompts, num_multimask_outputs)
    """
    masks, iou_pred = self.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
    )
    if multimask_output:
      return masks[:, 1:], iou_pred[:, 1:]
    else:
      return masks[:, :1], iou_pred[:, :1]


class MLP(nn.Module):
  hidden_dim: int
  output_dim: int
  num_layers: int

  @nn.compact
  def __call__(self, x):
    for i in range(self.num_layers - 1):
      x = nn.Dense(self.hidden_dim, name=f'layers.{i}')(x)
      x = nn.relu(x)
    x = nn.Dense(self.output_dim, name=f'layers.{self.num_layers - 1}')(x)
    return x


class OutputScaling(nn.Module):
  """Output scaling."""
  transformer_dim: int

  @nn.compact
  def __call__(self, x):
    x = nn.ConvTranspose(
        self.transformer_dim // 4, kernel_size=(2, 2), strides=(2, 2),
        transpose_kernel=True,
        name='0')(x)
    x = nn.LayerNorm(name='1')(x)
    x = nn.gelu(x, approximate=False)
    x = nn.ConvTranspose(
        self.transformer_dim // 8, kernel_size=(2, 2), strides=(2, 2),
        transpose_kernel=True,
        name='3')(x)
    x = nn.gelu(x, approximate=False)
    return x
