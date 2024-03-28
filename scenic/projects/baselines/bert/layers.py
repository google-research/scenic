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

"""BERT Layers."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import nn_layers

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def bert_truncated_normal_initializer():
  """TruncatedNormal(0.02) initializer."""

  def init(key, shape, dtype=jnp.float32):
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return jax.random.truncated_normal(key, -2, 2, shape, dtype) * 0.02

  return init


def sinusoidal_init(max_len: int = 2048,
                    min_scale: float = 1.0,
                    max_scale: float = 10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: Maximum possible length for the input.
      min_scale: Minimum frequency-scale in sine grating.
      max_scale: Maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)
  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    max_len: Maximum supported length
    posemb_init: Positional embedding initializer.
  """
  max_len: int
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.
    Args:
      inputs: input data.
    Returns:
      output: `[bs, timesteps, in_dim]`
    """
    # Inputs.shape is [batch_size, seq_len, emb_dim].
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=self.max_len)(None, pos_emb_shape,
                                                            None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    return inputs + pe


class Stem(nn.Module):
  """Stem for BERT.

  Attributes:
    vocab_size: Size of words/tokens vocabulary.
    type_vocab_size: Size of type vocabulary.
    hidden_size: Size of the hidden state of the output of model's stem.
    max_position_embeddings: The maximum sequence length that this model might
      ever be used with.
    embedding_width: Size of embedding
    dropout_rate: Dropout rate.
    dtype: JAX data type for activations.
  """
  vocab_size: int
  type_vocab_size: int
  hidden_size: int
  max_position_embeddings: int
  embedding_width: Optional[int] = None
  dropout_rate: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(
      self, input_word_ids: jnp.ndarray, input_type_ids: jnp.ndarray,
      input_mask: jnp.ndarray, *,
      train: bool) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    if input_word_ids.ndim != 2:
      raise ValueError('Input_word_ids should be of shape `[bs, l]` but it is '
                       f'{input_word_ids.shape}.')
    if input_type_ids.ndim != 2:
      raise ValueError('Input_type_ids should be of shape `[bs, l]` but it is '
                       f'{input_type_ids.shape}.')
    if input_mask.ndim != 2:
      raise ValueError('Input_mask should be of shape `[bs, l]` but it is '
                       f'{input_mask.shape}.')

    embedding_width = (
        self.embedding_width if self.embedding_width else self.hidden_size)

    word_embedding_layer = nn.Embed(
        num_embeddings=self.vocab_size,
        features=embedding_width,
        embedding_init=bert_truncated_normal_initializer(),
        name='word_embedding')
    x = word_embedding_layer(input_word_ids)
    x = x + nn.Embed(
        num_embeddings=self.type_vocab_size,
        features=embedding_width,
        embedding_init=bert_truncated_normal_initializer(),
        name='type_embedding')(
            input_type_ids)
    # NOTE: CLS token is added during pre-processing in the tokenizer.
    x = AddPositionEmbs(
        max_len=self.max_position_embeddings,
        posemb_init=bert_truncated_normal_initializer(),
        name='posembed_input')(
            x)
    x = nn.LayerNorm(dtype=self.dtype, name='embedding_norm')(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    if embedding_width != self.hidden_size:
      x = nn.Dense(
          self.hidden_size,
          kernel_init=bert_truncated_normal_initializer(),
          name='embedding_projection')(
              x)
    return x, word_embedding_layer.embedding


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    dtype: floating point type used in the layer.
    mlp_dim: hidden dimension of the multilayer perceptron.
    dropout_rate: dropout rate used in the hidden layer.
    kernel_init: weight matrix initializer.
    bias_init: bias vector initializer.
  """
  dtype: Any = jnp.float32
  mlp_dim: int = 2048
  dropout_rate: float = 0.1
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    """Applies Transformer MlpBlock module."""
    out_dim = inputs.shape[-1]
    x = nn.Dense(self.mlp_dim,
                 dtype=self.dtype,
                 kernel_init=self.kernel_init,
                 bias_init=self.bias_init)(inputs)
    x = nn_layers.IdentityLayer(name='gelu')(nn.gelu(x))
    x = nn.Dropout(rate=self.dropout_rate)(
        x, deterministic=not train)
    output = nn.Dense(out_dim,
                      dtype=self.dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)(x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=not train)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer/BERT encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    pre_norm: Whether to use PreLN, otherwise PostLN. For more detail, see
      https://arxiv.org/pdf/2002.04745.pdf.
    dtype: The dtype of the computation (default: float32).

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  pre_norm: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, input_mask: Optional[jnp.ndarray],
               deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      input_mask: Input mask, used for text input.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3

    # pre-attention-layer-normalization
    x = nn.LayerNorm(dtype=self.dtype)(inputs) if self.pre_norm else inputs
    attention_mask = input_mask[:, None, None, :] * jnp.ones(
        [1, 1, x.shape[1], 1])
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate)(
            x, x, mask=attention_mask, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = x + inputs
    if not self.pre_norm:  # post-attention-layer-normalization
      # Normalized x is used for residual connection.
      x = nn.LayerNorm(dtype=self.dtype)(x)

    # MLP block.
    if self.pre_norm:  # pre-mlp-layer-normalization
      # Do not overwrite x because it will be used for residual connection.
      y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        dtype=self.dtype,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y if self.pre_norm else x, train=not deterministic)
    y = y + x
    # post-mlp-layer-normalization
    if not self.pre_norm:
      y = nn.LayerNorm(dtype=self.dtype)(y)
    return y


class BERTEncoder(nn.Module):
  """BERT encoder.

    Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    pre_norm: Whether to use PreLN in encoder layers, otherwise PostLN.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  pre_norm: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, input_mask: jnp.ndarray, *, train: bool):

    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          pre_norm=self.pre_norm,
          name=f'encoderblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
              x, input_mask=input_mask, deterministic=not train)

    if self.pre_norm:
      x = nn.LayerNorm(name='encoder_norm')(x)
    return x


class ClassificationHead(nn.Module):
  """Head used for classification with BERT.

  Attributes:
    num_outputs: Number of output classes.
    hidden_sizes: Size of hidden units in additional projections in the head.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    nonlinearity: Nonlinearity, ReLU by default.
    dtype: Model dtype.
  """
  num_outputs: int
  hidden_sizes: Union[int, Tuple[int, ...]]
  kernel_init: Initializer = initializers.lecun_normal()
  bias_init: Initializer = initializers.zeros
  nonlinearity: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    # Get CLS token:
    x = x[:, 0]
    hidden_sizes = self.hidden_sizes
    if isinstance(hidden_sizes, int):
      hidden_sizes = [hidden_sizes]
    for num_hid in hidden_sizes:
      # These intermediate layers are only used in BERT.
      x = nn.Dense(
          num_hid,
          kernel_init=bert_truncated_normal_initializer(),
          bias_init=self.bias_init)(
              x)
      x = self.nonlinearity(x)

    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_outputs,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='output_projection')(
            x)
    return x


class MaskedLanguageModelHead(nn.Module):
  """Head used for masked language modelling in BERT.

  Attributes:
    dtype: Data type.
  """
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, masked_lm_positions: jnp.ndarray,
               word_embeddings: jnp.ndarray, *, train: bool) -> jnp.ndarray:

    batch_size, length, hidden_size = x.shape
    x = x.reshape((-1, hidden_size))

    offsets = np.arange(batch_size)[:, None] * length
    masked_lm_positions = masked_lm_positions + offsets
    masked_lm_positions = masked_lm_positions.ravel()

    x = jnp.take(x, masked_lm_positions, axis=0)
    x = x.reshape((batch_size, -1, hidden_size))

    vocab_size, embedding_width = word_embeddings.shape
    kernel_init = bert_truncated_normal_initializer()

    x = nn.Dense(embedding_width, kernel_init=kernel_init, dtype=self.dtype)(x)
    x = nn.gelu(x)
    x = nn.LayerNorm(dtype=self.dtype)(x)
    x = jnp.einsum('ijk,lk->ijl', x, word_embeddings)
    x = x + self.param('embedding_bias', nn.initializers.zeros,
                       (1, 1, vocab_size), self.dtype)
    return x
