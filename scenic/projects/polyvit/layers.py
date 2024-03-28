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

"""Layers used in PolyVit."""

import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.polyvit import polyvit_base_model
from scenic.projects.vivit import model as vivit_model


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def sinusoidal_init(max_len: int = 2048,
                    min_scale: float = 1.0,
                    max_scale: float = 10000.0) -> Initializer:
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32) -> jnp.ndarray:
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


def get_bottleneck_representation(x, pooling_type):
  """Bottleneck representation.

  Args:
      x: input tensor.
      pooling_type: type of the classifier layer. Options are 'gap', 'gmp',
      'gsp', 'token'.

  Returns:
      bottleneck tensor.
  """

  if pooling_type in ('token', '0'):
    x = x[:, 0]
  elif pooling_type in ('gap', 'gmp', 'gsp'):
    fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[pooling_type]
    x = fn(x, axis=1)

    x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))
  else:
    raise ValueError(
        "Pooling type should be in ['gap', 'gmp', 'gsp', 'token'].")

  return nn_layers.IdentityLayer(name='bottleneck')(x)


def get_droplayer_p(layer, num_layers, stochastic_droplayer_rate):
  """Stochastic drop-layer probability.

  Args:
      layer: Layer index.
      num_layers: Total number of layers.
      stochastic_droplayer_rate: Probability of dropping a layer linearly grows
        from 0 to the provided value.
  Returns:
      Probability.
  """

  if stochastic_droplayer_rate is None:
    return None

  return (layer / max(num_layers - 1, 1)) * stochastic_droplayer_rate


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    max_len: maximal length of the input sequence.
    posemb_init: positional embedding initializer.
  """

  max_len: int = 2048
  posemb_init: Optional[Initializer] = None

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               inputs_positions: Any = None) -> jnp.ndarray:
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=self.max_len)(None, pos_emb_shape,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                            None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


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
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    output = nn.Dense(out_dim,
                      dtype=self.dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)(x)
    output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=not train)
    return output


class DynamicMultiHeadAttention(nn.Module):
  """Customized dynamic multi-head attention for scenic.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    qkv_features: Dimension of the key, query, and value.
    out_features: Dimension of the last projection.
    dropout_rate: Dropout rate.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    kernel_init: Initializer for the kernel of the Dense layers.
    bias_init: Initializer for the bias of the Dense layers.
    use_bias: Whether pointwise QKV dense transforms use bias.
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    dtype: the dtype of the computation (default: float32).
  """
  num_heads: int
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  dropout_rate: float = 0.
  broadcast_dropout: bool = False
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  use_bias: bool = True
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs_q: jnp.ndarray,
               inputs_kv: Optional[jnp.ndarray],
               dataset: str,
               *,
               pos_emb_q: Optional[jnp.ndarray] = None,
               pos_emb_k: Optional[jnp.ndarray] = None,
               pos_emb_v: Optional[jnp.ndarray] = None,
               attention_bias: Optional[jnp.ndarray] = None,
               attention_bias_kv: Optional[jnp.ndarray] = None,
               deterministic: bool = False) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: Input queries of shape  `[bs, ..., len_q, features]`.
      inputs_kv: Key/values of shape `[bs, ..., len_k, features]` or None for
        self-attention, inn which case key/values will be derived from inputs_q.
      dataset: Current dataset.
      pos_emb_q: Positional embedding to be added to the query.
      pos_emb_k: Positional embedding to be added to the key.
      pos_emb_v: Positional embedding to be added to the value.
      attention_bias: Full attention bias. Should be broadcastable to:
        inputs_q.shape[:-2] + (num_heads, len_q, len_k).
      attention_bias_kv: Attention bias for keys independent of queries which
        has shape (bs, ..., len_k).
      deterministic: Run deterministically or with dropout.

    Returns:
      Output of shape `[bs, ..., features]`.
    """
    if inputs_kv is None:
      inputs_kv = inputs_q

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    def add_positional_emb(x, pos):
      return x + pos if pos is not None else x

    query, key, value = (add_positional_emb(inputs_q, pos_emb_q),
                         add_positional_emb(inputs_kv, pos_emb_k),
                         add_positional_emb(inputs_kv, pos_emb_v))

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)
    # Project inputs_q to multi-headed q/k/v.
    # Dimensions are then [..., l, n_heads, n_features_per_head].
    query, key, value = (dense(name='query')(query),
                         dense(name='key')(key),
                         dense(name='value')(value))

    # pylint: disable=too-many-function-args
    attn_kwargs = {}
    if attention_bias_kv is not None:
      # Not necessarily supported by all underlying functions.
      attn_kwargs['bias_kv'] = attention_bias_kv
    if not deterministic and self.dropout_rate > 0:
      attn_kwargs['dropout_rng'] = self.make_rng('dropout')

    attention_fn = attention_layers.dot_product_attention
    x = attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision,
        **attn_kwargs)
    # pylint: enable=too-many-function-args

    # Back to the original inputs dimensions.
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(
            x)

    return out


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  def get_droplayer_mask(self, x: jnp.ndarray, deterministic: bool,
                         droplayer_p: Optional[float]) -> jnp.ndarray:
    """Generate the drop-layer mask.

    Args:
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.
      droplayer_p: Probability of dropping a layer.

    Returns:
      Droplayer mask.
    """
    if not deterministic and droplayer_p is not None:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(self.make_rng('dropout'), droplayer_p, shape)
    else:
      return 0.0  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool,
               droplayer_p: Optional[float], dataset: str) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).
      droplayer_p: Probability of dropping a layer.
      dataset: Current dataset.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = x * (1.0 -
             self.get_droplayer_mask(x, deterministic, droplayer_p)) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        dtype=self.dtype,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, train=not deterministic)

    return y * (1.0 -
                self.get_droplayer_mask(x, deterministic, droplayer_p)) + x


class Tokenizer2D(nn.Module):
  """Tokenizer for 2D inputs (e.g., images).

  Attributes:
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    add_cls_token: Whether to add CLS token or not.
    dtype: JAX data type for activations.
  """
  patches: ml_collections.ConfigDict
  hidden_size: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_cls_token: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, dataset: str,
               stochastic_droplayer_rate: Optional[float]) -> jnp.ndarray:
    if x.ndim != 4:
      raise ValueError(
          f'Input shape should be `[bs, h, w, c]` but it is {x.shape}.')
    fh, fw = self.patches.size
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    if self.add_cls_token:
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            x)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    for lyr in range(self.num_layers):
      droplayer_p = get_droplayer_p(lyr, self.num_layers,
                                    stochastic_droplayer_rate)
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
              x,
              deterministic=not train,
              droplayer_p=droplayer_p,
              dataset=dataset)

    return x


class Tokenizer3D(nn.Module):
  """Tokenizer for 3D inputs (e.g., videos).

  Attributes:
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    kernel_init_method: Method for initializing the kernel. Options are
      `central_frame_initializer` (which is the best performing one in ViViT),
      `average_frame_initializer`, and None (using flax default_kernel_init).
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    add_cls_token: whether to add CLS token or not.
    dtype: JAX data type for activations.
  """
  patches: ml_collections.ConfigDict
  hidden_size: int
  kernel_init_method: Optional[str]
  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_cls_token: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, dataset: str,
               stochastic_droplayer_rate: Optional[float]) -> jnp.ndarray:
    if x.ndim != 5:
      raise ValueError(
          f'Input shape should be `[bs, t, h, w, c]` but it is {x.shape}.')
    x = vivit_model.embed_3d_patch(
        x,
        self.patches,
        self.hidden_size,
        kernel_init_method=self.kernel_init_method)
    n, t, h, w, c = x.shape
    x = jnp.reshape(x, [n, t * h * w, c])
    if self.add_cls_token:
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            x)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    for lyr in range(self.num_layers):
      droplayer_p = get_droplayer_p(lyr, self.num_layers,
                                    stochastic_droplayer_rate)
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
              x,
              deterministic=not train,
              droplayer_p=droplayer_p,
              dataset=dataset)

    return x


class Tokenizer(nn.Module):
  """Unified Tokenizer class.

  Attributes:
    config: Tokenizer config.
    dtype: JAX data type for activations.
  """

  config: ml_collections.ConfigDict
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, dataset: str,
               modality: Optional[str],
               stochastic_droplayer_rate: Optional[float]) -> jnp.ndarray:

    if modality == polyvit_base_model.Modality.IMAGE:
      return Tokenizer2D(
          self.config.image.patches,
          self.config.hidden_size,
          self.config.mlp_dim,
          self.config.num_layers,
          self.config.num_heads,
          dropout_rate=self.config.dropout_rate,
          attention_dropout_rate=self.config.attention_dropout_rate,
          add_cls_token=self.config.add_cls_token,
          dtype=self.dtype,
          name='tokenizer2d')(
              x,
              train=train,
              stochastic_droplayer_rate=stochastic_droplayer_rate,
              dataset=dataset)
    elif modality == polyvit_base_model.Modality.VIDEO:
      return Tokenizer3D(
          self.config.video.patches,
          self.config.hidden_size,
          self.config.video.kernel_init_method,
          self.config.mlp_dim,
          self.config.num_layers,
          self.config.num_heads,
          dropout_rate=self.config.dropout_rate,
          attention_dropout_rate=self.config.attention_dropout_rate,
          add_cls_token=self.config.add_cls_token,
          dtype=self.dtype,
          name='tokenizer3d')(
              x,
              train=train,
              stochastic_droplayer_rate=stochastic_droplayer_rate,
              dataset=dataset)
    elif modality == polyvit_base_model.Modality.AUDIO:
      return Tokenizer2D(
          self.config.audio.patches,
          self.config.hidden_size,
          self.config.mlp_dim,
          self.config.num_layers,
          self.config.num_heads,
          dropout_rate=self.config.dropout_rate,
          attention_dropout_rate=self.config.attention_dropout_rate,
          add_cls_token=self.config.add_cls_token,
          dtype=self.dtype,
          name='tokenizer_spec')(
              x,
              train=train,
              stochastic_droplayer_rate=stochastic_droplayer_rate,
              dataset=dataset)
    else:
      raise NotImplementedError(f'Modality {modality} is not supported yet.')


class PolyViTEncoder(nn.Module):
  """PolyViT encoder.

    Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_tokenizer_layers: Number of tokenizer layers.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_tokenizer_layers: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, dataset: str,
               stochastic_droplayer_rate: Optional[float]):

    for lyr in range(self.num_tokenizer_layers, self.num_layers):
      droplayer_p = get_droplayer_p(lyr, self.num_layers,
                                    stochastic_droplayer_rate)
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
              x,
              deterministic=not train,
              droplayer_p=droplayer_p,
              dataset=dataset)

    x = nn.LayerNorm(name='encoder_norm')(x)

    return x


class ClassificationHead(nn.Module):
  """Defines a fully connected neural network.

  The model assumes the input data has shape
  [batch_size_per_device, *input_shape] where input_shape may be of arbitrary
  rank. The model flatten the input before applying a dense layer.

  Attributes:
    num_outputs: Number of output classes.
    hid_sizes: Size of hidden units in each layer.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    output_proj_zero_init: Whether to initialize the output projection with
      zeros.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: Model dtype.
  """
  num_outputs: int
  hid_sizes: Union[Tuple[int, ...], int] = ()
  kernel_init: Initializer = initializers.lecun_normal()
  bias_init: Initializer = initializers.zeros
  output_proj_zero_init: bool = False
  classifier: str = 'gap'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:

    del train
    hid_sizes = self.hid_sizes
    if isinstance(hid_sizes, int):
      hid_sizes = [hid_sizes]

    x = get_bottleneck_representation(x, self.classifier)

    for num_hid in hid_sizes:
      x = nn.Dense(
          num_hid, kernel_init=self.kernel_init, bias_init=self.bias_init)(
              x)
      x = nn.relu(x)

    # Head.
    x = nn_layers.IdentityLayer(name='pre_logits')(x)

    if self.output_proj_zero_init:
      output_proj_kernel_init = nn.initializers.zeros
      output_proj_bias_init = nn.initializers.zeros
    else:
      output_proj_kernel_init = self.kernel_init
      output_proj_bias_init = self.bias_init

    x = nn.Dense(
        self.num_outputs,
        kernel_init=output_proj_kernel_init,
        bias_init=output_proj_bias_init,
        name='output_projection')(
            x)
    return x


class FewshotHead(nn.Module):
  """Head used for fewshot metrics.

  There are no trainable parameters in this module.

  Attributes:
    pooling_type: type of the bottleneck. Options are 'gap', 'gmp', 'gsp',
    'token'.
  """
  pooling_type: str = 'gap'

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:

    del train

    x = get_bottleneck_representation(x, self.pooling_type)

    return x
