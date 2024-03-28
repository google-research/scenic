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

"""Token Turing Machines.

https://arxiv.org/abs/2211.09119
"""

from typing import Tuple

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp

from scenic.model_lib.layers import attention_layers
from scenic.projects.baselines import mixer
from scenic.projects.baselines import vit
from scenic.projects.token_learner import model as token_learner_model


class TokenLearnerMHA(nn.Module):
  """TokenLearner module using MHA.

  Attributes:
    num_tokens: Number of tokens to generate.
    num_heads: Number of heads to use for the dot product attention.
  """
  num_tokens: int
  num_heads: int = 8

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies TokenLearner-mha to the inputs.

    Args:
      inputs: Inputs of shape `[bs, hw, c]`.

    Returns:
      Output of shape `[bs, num_tokens, c]`.
    """
    bs, _, d = inputs.shape

    token_init = nn.initializers.normal(stddev=0.02)
    tokens = self.param('tokens', token_init, (1, self.num_tokens, d))
    tokens = jnp.broadcast_to(tokens, (bs, self.num_tokens, d))
    return nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, deterministic=True, name='attn')(
            inputs_q=tokens, inputs_kv=inputs)


class TokenAddEraseWrite(nn.Module):
  """Token write operations motivated by the `write' in Neural Turing Machines.

  Instead of directly using the token summarization (with TokenLearner), it uses
  a similar but different mechanism to (soft-)select memory elements to zero out
  and write to them. This can be used as an alternative write operation in the
  TTM, particularly when the memory size is huge.
  """

  num_tokens: int = 8
  bottleneck_dim: int = 64
  dropout_rate: float = 0.

  @nn.compact
  def __call__(self,
               memory: jnp.ndarray,
               control_inputs: jnp.ndarray,
               training: bool = False) -> jnp.ndarray:

    selected = nn.LayerNorm()(memory)

    selected = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        out_dim=self.num_tokens,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu)(
            selected, deterministic=not training)

    selected = jnp.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, hw].
    selected = jax.nn.softmax(selected, axis=-1)

    et = nn.LayerNorm()(control_inputs)
    et = jnp.transpose(et, [0, 2, 1])  # Shape: [bs, c, hw].
    et = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        out_dim=self.num_tokens,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu)(
            et, deterministic=not training)  # Shape: [bs, c, n_token].
    et = jnp.transpose(et, [0, 2, 1])  # Shape: [bs, n_token, c].

    et = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu)(
            et, deterministic=not training)

    wet = jnp.expand_dims(selected, -1) * jnp.expand_dims(
        et, 2)  # Shape: [bs, n_token, hw, c].
    wet = 1 - wet
    wet = jnp.prod(wet, axis=1)  # Shape: [bs, hw, c].

    output = memory * wet

    at = nn.LayerNorm()(control_inputs)
    at = jnp.transpose(at, [0, 2, 1])  # Shape: [bs, c, hw].
    at = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        out_dim=self.num_tokens,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu)(
            at, deterministic=not training)  # Shape: [bs, c, n_token].
    at = jnp.transpose(at, [0, 2, 1])  # Shape: [bs, n_token, c].

    at = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu)(
            at, deterministic=not training)

    wat = jnp.expand_dims(selected, -1) * jnp.expand_dims(
        at, 2)  # Shape: [bs, n_token, hw, c].
    wat = 1 - wat
    wat = jnp.mean(wat, axis=1)  # Shape: [bs, hw, c].

    output += wat

    return output  # Shape: [bs, hw, c]


class TokenTuringMachineUnit(nn.Module):
  """One Token Turing Machine unit.

  This implements the operations in a TTM (https://arxiv.org/abs/2211.09119) at
  each step.

  Attributes:
    process_size: Number of tokens for the processing unit to process.
    memory_size: Number of memory tokens.
    memory_mode: Specifies the token summarization method to use. Supports
      'TL', 'TL-MHA', or 'TL-AddErase'.
    processing_unit: Specifies which processing unit module to use. Supports
      'transformer', 'mixer', or 'mlp'.
    num_layers: Number of layers in the processing unit.
    mlp_dim: MLP dim size in the processing unit.
    num_heads: Number of heads in the processing unit.
    use_positional_embedding: Whether to use positional embeddings for the
      memory read/write.
    dropout_rate: Dropout rate.
  """
  process_size: int = 8
  memory_size: int = 64
  memory_mode: str = 'TL'
  processing_unit: str = 'transformer'
  num_layers: int = 1
  mlp_dim: int = 512
  num_heads: int = 12
  use_positional_embedding: bool = False
  dropout_rate: float = 0.

  @nn.compact
  def __call__(self,
               memory_tokens: jnp.ndarray,
               input_tokens: jnp.ndarray,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies Token Turing Machine unit.

    Args:
      memory_tokens: Inputs of shape `[bs, memory_size, c]`.
      input_tokens: Inputs of shape `[bs, n_token, c]`.
      train: Weather we are in the training mode.

    Returns:
      Tuple of shape `([bs, memory_size, c], [bs, process_size, c])`.
    """
    all_tokens = jnp.concatenate([memory_tokens, input_tokens], axis=1)

    if self.use_positional_embedding:
      all_tokens = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='read_pos_embed')(all_tokens)

    if self.memory_mode == 'TL' or self.memory_mode == 'TL-AddErase':
      all_tokens = token_learner_model.TokenLearnerModuleV11(
          self.process_size,
          dropout_rate=self.dropout_rate)(all_tokens, deterministic=not train)
    elif self.memory_mode == 'TL-MHA':
      all_tokens = TokenLearnerMHA(self.process_size)(all_tokens)

    if self.processing_unit == 'transformer':
      output_tokens = vit.Encoder(
          num_layers=self.num_layers,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.dropout_rate,
          )(all_tokens, train=train)
    elif self.processing_unit == 'mixer':
      output_tokens = all_tokens
      for _ in range(self.num_layers):
        output_tokens = mixer.MixerBlock(
            channels_mlp_dim=self.mlp_dim,
            sequence_mlp_dim=128,
            dropout_rate=self.dropout_rate,
            stochastic_depth=0.,
            layer_scale=None)(
                output_tokens, deterministic=not train)
      output_tokens = nn.LayerNorm()(output_tokens)
    elif self.processing_unit == 'mlp':
      output_tokens = all_tokens
      for _ in range(self.num_layers):
        output_tokens = nn.LayerNorm()(output_tokens)
        output_tokens = attention_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu)(
                output_tokens, deterministic=not train)
      output_tokens = nn.LayerNorm()(output_tokens)
    else:
      raise ValueError(f'Unknown processing unit {self.processing_unit}.')
    mem_out_tokens = jnp.concatenate(
        [memory_tokens, input_tokens, output_tokens], axis=1)

    if self.use_positional_embedding:
      mem_out_tokens = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='write_pos_embed')(mem_out_tokens)

    if self.memory_mode == 'TL':
      mem_out_tokens = token_learner_model.TokenLearnerModuleV11(
          self.memory_size, dropout_rate=self.dropout_rate)(
              mem_out_tokens, deterministic=not train)
    elif self.memory_mode == 'TL-MHA':
      mem_out_tokens = TokenLearnerMHA(self.memory_size)(mem_out_tokens)
    elif self.memory_mode == 'TL-AddErase':
      mem_out_tokens = TokenAddEraseWrite()(memory_tokens, output_tokens, train)

    return (mem_out_tokens, output_tokens)


class TokenTuringMachineSimpleUnit(nn.Module):
  """Token Turing Machine unit, simplified.

  Instead of implementing the memory read/write with TokenLearner, it directly
  relies on a Transformer processing unit to maintain the memory.

  TokenLearner is still used to reduce the number of input tokens.

  Attributes:
    process_size: Number of tokens for the Transformer to process
    memory_size: The number of memory tokens to maintain.
    processing_unit: Specifies which processing unit module to use.
    num_layers: Number of layers in the processing unit.
    mlp_dim: MLP dim size in the processing unit.
    num_heads: Number of heads in the processing unit.
    dropout_rate: Dropout rate.
  """
  process_size: int = 16
  memory_size: int = 96
  processing_unit: str = 'transformer'
  num_layers: int = 1
  mlp_dim: int = 512
  num_heads: int = 12
  dropout_rate: float = 0.

  @nn.compact
  def __call__(self,
               memory_tokens: jnp.ndarray,
               input_tokens: jnp.ndarray,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies Token Turing Machine unit.

    Args:
      memory_tokens: Inputs of shape `[bs, memory_size, c]`.
      input_tokens: Inputs of shape `[bs, n_token, c]`.
      train: Weather we are in the training mode.

    Returns:
      Tuple of shape `([bs, memory_size, c], [bs, n_token, c])`.
    """

    input_tokens = token_learner_model.TokenLearnerModuleV11(
        self.process_size,
        dropout_rate=self.dropout_rate)(input_tokens, deterministic=not train)

    all_tokens = jnp.concatenate([memory_tokens, input_tokens], axis=1)

    if self.processing_unit == 'transformer':
      output_tokens = vit.Encoder(
          num_layers=self.num_layers,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.dropout_rate,
          )(all_tokens, train=train)
    elif self.processing_unit == 'mixer':
      output_tokens = all_tokens
      for _ in range(self.num_layers):
        output_tokens = mixer.MixerBlock(
            channels_mlp_dim=self.mlp_dim,
            sequence_mlp_dim=128,
            dropout_rate=self.dropout_rate,
            stochastic_depth=0.,
            layer_scale=None)(
                output_tokens, deterministic=not train)
      output_tokens = nn.LayerNorm()(output_tokens)
    elif self.processing_unit == 'mlp':
      output_tokens = all_tokens
      for _ in range(self.num_layers):
        output_tokens = nn.LayerNorm()(output_tokens)
        output_tokens = attention_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu)(
                output_tokens, deterministic=not train)
      output_tokens = nn.LayerNorm()(output_tokens)

    mem_out_tokens = output_tokens[:, self.process_size:, :]
    output_tokens = output_tokens[:, :self.process_size, :]

    return (mem_out_tokens, output_tokens)


class TokenTuringMachineEncoder(nn.Module):
  """Token Turing Machine main model encoder.

  It implements https://arxiv.org/abs/2211.09119. It essentially repeats
  TokenTuringMachineUnit for the number of steps (of the input tensor).

  This version is for the training and inference with a fixed shaped, static
  input tensor. One will need to modify/extend this module together with the
  data pipeline for the streaming inference implementation.

  Attributes:
    process_size: Number of tokens for the Transformer to process.
    memory_size: The number of memory tokens in the TTM.
    memory_mode: Specifies the token summarization method to use. Supports
      'TL', 'TL-MHA', or 'TL-AddErase'.
    processing_unit: Specifies which processing unit module to use. Supports
      'transformer', 'mixer', or 'mlp'.
    num_layers: Number of layers in the processing unit.
  """
  process_size: int = 8
  memory_size: int = 64
  memory_mode: str = 'TL'
  processing_unit: str = 'transformer'
  num_layers: int = 4
  dropout_rate: float = 0.

  def setup(self):
    self.unit_function = TokenTuringMachineUnit(
        process_size=self.process_size,
        memory_size=self.memory_size,
        num_layers=self.num_layers,
        memory_mode=self.memory_mode,
        processing_unit=self.processing_unit,
        dropout_rate=self.dropout_rate)

  def __call__(self,
               input_tokens: jnp.ndarray,
               train: bool = False) -> jnp.ndarray:
    """Applies Token Turing Machine model.

    Args:
      input_tokens: Inputs of shape `[bs, num_steps, n_tokens, c]`.
      train: Weather we are in the training mode.

    Returns:
      Tensor of shape `[bs, num_steps, process_size, c]`.
    """
    bs, ns, _, c = input_tokens.shape

    output_tokens_list = []
    memory_tokens = jnp.zeros([bs, self.memory_size, c])

    for i in range(ns):
      step_tokens = input_tokens[:, i, :, :]

      logging.info('step tokens: %s', step_tokens.shape)
      memory_tokens, output_tokens = self.unit_function(
          memory_tokens, step_tokens, train=train)

      logging.info('output_tokens: %s', output_tokens.shape)
      logging.info('memory_tokens: %s', memory_tokens.shape)

      output_tokens = jnp.expand_dims(output_tokens, axis=1)
      output_tokens_list.append(output_tokens)

    output_tokens = jnp.concatenate(output_tokens_list, axis=1)

    return output_tokens
