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

"""Prompt Adapter."""

import flax.linen as nn
import jax.numpy as jnp
from scenic.projects.baselines.segment_anything.modeling import transformer


class PromptAdaptor(nn.Module):
  """Prompt Adaptor to adapt visual feature with location information."""

  depth: int = 2
  input_embedding_dim: int = 256
  transformer_dim: int = 256
  num_outputs: int = 32
  head_hidden_dim: int = 2048
  output_dim: int = 1024

  def setup(self):
    self.output_tokens = self.param(
        'output_tokens.weight',
        nn.initializers.normal(stddev=1.0),
        (self.num_outputs, self.transformer_dim),
    )

    self.output_hypernework_mlps = [
        MLP(hidden_dim=self.head_hidden_dim,
            output_dim=self.output_dim, num_layers=1,
            name=f'output_hypernetworks_mlps.{i}')
        for i in range(self.num_outputs)
    ]
    self.image_projection_mlp = MLP(
        hidden_dim=self.head_hidden_dim,
        output_dim=self.transformer_dim,
        num_layers=1,
        name='image_projection_mlp',
    )
    if self.input_embedding_dim != self.transformer_dim:
      self.prompt_embedding_projection_mlp = MLP(
          hidden_dim=self.head_hidden_dim,
          output_dim=self.transformer_dim,
          num_layers=1,
          name='prompt_embedding_projection_mlp',
      )
      self.image_pe_projection_mlp = MLP(
          hidden_dim=self.head_hidden_dim,
          output_dim=self.transformer_dim,
          num_layers=1,
          name='image_pe_projection_mlp',
      )
    else:
      self.prompt_embedding_projection_mlp = None
      self.image_pe_projection_mlp = None

    self.transformer = transformer.TwoWayTransformer(
        depth=self.depth,
        embedding_dim=self.transformer_dim,
        name='transformer'
    )

    self.dense_output_mlp = MLP(
        hidden_dim=self.head_hidden_dim,
        output_dim=self.output_dim,
        num_layers=1,
        name='dense_output_mlp',
    )

  def predict_outputs(
      self,
      image_embeddings,
      image_pe,
      sparse_prompt_embeddings,
      dense_prompt_embeddings,
  ):
    """Predict masks for a single image.

    Args:
      image_embeddings: (H, W, embed_dim)
      image_pe: (H, W, embed_dim)
      sparse_prompt_embeddings: (num_prompts, num_points, embed_dim)
      dense_prompt_embeddings: (num_prompts, H, W, embed_dim)

    Returns:
      hs: (num_prompts, num_outputs, transformer_dim)
      src: (num_prompts, h, w, transformer_dim)
    """
    if self.prompt_embedding_projection_mlp is not None:
      sparse_prompt_embeddings = self.prompt_embedding_projection_mlp(
          sparse_prompt_embeddings
      )
      dense_prompt_embeddings = self.prompt_embedding_projection_mlp(
          dense_prompt_embeddings
      )
    if self.image_pe_projection_mlp is not None:
      image_pe = self.image_pe_projection_mlp(image_pe)

    output_tokens = self.output_tokens  # (num_outputs, transformer_dim)
    num_prompts = sparse_prompt_embeddings.shape[0]
    output_tokens = jnp.broadcast_to(
        output_tokens[None],
        (num_prompts, self.num_outputs, self.transformer_dim),
    )
    tokens = jnp.concatenate(
        [output_tokens, sparse_prompt_embeddings],
        axis=1,
    )  # (num_prompts, num_outputs + num_points, embed_dim)

    src = jnp.repeat(
        image_embeddings[None], tokens.shape[0], axis=0
    )  # (num_prompts, H, W, D)
    src = self.image_projection_mlp(src) + dense_prompt_embeddings
    pos_src = jnp.repeat(
        image_pe[None], tokens.shape[0], axis=0
    )  # (num_prompts, H, W, D)
    num_prompts, h, w, d = src.shape

    hs, src = self.transformer(src, pos_src, tokens)
    tokens_out = hs[:, : self.num_outputs, :]

    hyper_in_list = []
    for i in range(self.num_outputs):
      hyper_in_list.append(
          self.output_hypernework_mlps[i](
              tokens_out[:, i, :]
          )  # (num_prompts, d)
      )
    hyper_in = jnp.stack(hyper_in_list, axis=1)  # (num_prompts, num_outputs, d)

    src = src.reshape(num_prompts, h, w, d)

    src = self.dense_output_mlp(src)

    return hyper_in, src

  @nn.compact
  def __call__(
      self,
      image_embeddings,
      image_pe,
      sparse_prompt_embeddings,
      dense_prompt_embeddings,
  ):
    """Forward model for a single image.

    Args:
      image_embeddings: (H, W, 3)
      image_pe: (H, W, D)
      sparse_prompt_embeddings: (num_prompts, num_points, embed_dim)
      dense_prompt_embeddings: (num_prompts, H, W, embed_dim)

    Returns:
    """
    hs, src = self.predict_outputs(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
    )
    return hs, src


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
