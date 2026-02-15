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

"""Topological attention core modules for Flax."""

import functools
from typing import Any, Callable, Optional, Tuple

from flax.linen.initializers import normal
from flax.linen.initializers import ones
from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from scenic.model_lib.layers import random_walk_utils


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

normal_initialiser = normal(stddev=1.0, dtype=jnp.float32)


def decaying_init_func(key, nb_heads, max_steps, decay_rate):
  """Initialise a decaying function for the Taylor params."""
  #  init function needs a PRNG key argument. Implicitly assumed in self.param.
  del key
  return jnp.asarray(
      [jnp.exp(-jnp.arange(max_steps) * decay_rate) for _ in range(nb_heads)]
  )


def topological_dot_product_attention_weights(
    query: Array,
    key: Array,
    topological_params: Array,
    top_type: str = 'toeplitz',
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Dtype = jnp.float32,
    precision: Optional[lax.Precision] = None,
    nb_x_patches: int = 0,
    nb_y_patches: int = 0,
    kernel_type: str = 'relu'):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    topological_params: tensor defining parameters of topological masking.
    top_type: type of the topological masking used. 'toeplitz', 'heat', 'taylor
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    nb_x_patches: number of patches in a fixed column.
    nb_y_patches: number of patches in a fixed row.
    kernel_type: type of the attention kernel used.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], ('q, k batch dims must match.')
  assert query.shape[-2] == key.shape[-2], ('q, k num_heads must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)

  if kernel_type == 'relu':
    query = jax.nn.relu(query) + 1e-8
    key = jax.nn.relu(key) + 1e-8

  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # ~~~~~~~ ENUMERATING ALL TYPE OF TOPOLOGICAL MASKING. ~~~~~~~~
  if top_type == 'toeplitz':  # block toeplitz mask
    grid_i = jnp.arange(nb_x_patches)
    grid_j = jnp.arange(nb_y_patches)
    dist_index = (grid_i[:, None, None, None] - grid_i[None, None, :, None] +
                  nb_x_patches) * 2 * nb_y_patches + grid_j[
                      None, :, None, None] - grid_j[None, None,
                                                    None, :] + nb_y_patches

    # Choose Toeplitz type.
    # dist_index = jnp.abs(
    #    grid_i[:, None, None, None] - grid_i[None, None, :, None]
    #) + jnp.abs(grid_j[None, :, None, None] - grid_j[None, None, None, :])
    dist_index = dist_index.reshape((-1,))
    top_mask = jnp.take(topological_params, dist_index, axis=1)
    top_mask = top_mask.reshape((
        topological_params.shape[0],
        nb_x_patches * nb_y_patches,
        nb_x_patches * nb_y_patches,
    ))
    top_mask = jnp.pad(top_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)
    top_mask = jnp.pad(top_mask, [[0, 0], [0, 0], [1, 0]], constant_values=1.0)
    top_mask = top_mask[None, ...]

  elif top_type == 'heat':  # heat kernel mask
    adj_matrix = random_walk_utils.get_grid_graph_adjacency_matrix(
        nb_x_patches, nb_y_patches, normalised=True
    )
    top_mask = jscipy.linalg.expm(
        jnp.asarray(
            [adj_matrix * topological_params[i] for i in range(query.shape[-2])]
        )
    )
    top_mask = jnp.pad(top_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)
    top_mask = jnp.pad(top_mask, [[0, 0], [0, 0], [1, 0]], constant_values=1.0)
    top_mask = top_mask[None, ...]

  elif top_type == 'taylor':  # learnable taylor expansion
    max_steps = 100
    adj_matrix = random_walk_utils.get_grid_graph_adjacency_matrix(
        nb_x_patches, nb_y_patches, normalised=True
    )
    adj_powers = jnp.asarray([
        jnp.linalg.matrix_power(adj_matrix, i)
        * topological_params[:, i][..., None, None]
        for i in range(max_steps)
    ])
    top_mask = jnp.sum(adj_powers, axis=0)
    top_mask = jnp.pad(top_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)
    top_mask = jnp.pad(top_mask, [[0, 0], [0, 0], [1, 0]], constant_values=1.0)
    top_mask = top_mask[None, ...]

  elif (
      top_type == 'learned_features'
  ):  # endow each token with a learned f mask:
    top_mask = jnp.asarray([
        jnp.matmul(topological_params[i], topological_params[i].T)
        for i in range(query.shape[-2])
    ])
    top_mask = jnp.pad(top_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)
    top_mask = jnp.pad(top_mask, [[0, 0], [0, 0], [1, 0]], constant_values=1.0)
    top_mask = top_mask[None, ...]

  elif top_type == 'jlt_taylor':  #  TE jlt'd to reduce dimensionality
    max_steps = 100
    jlt_mat = topological_params[0]
    adj_coefficients = topological_params[1]
    adj_matrix = random_walk_utils.get_grid_graph_adjacency_matrix(
        nb_x_patches, nb_y_patches, normalised=True
    )
    adj_powers = jnp.asarray([
        jnp.linalg.matrix_power(adj_matrix, i)
        * adj_coefficients[:, i][..., None, None]
        for i in range(max_steps)
    ])
    phi = jnp.sum(adj_powers, axis=0)
    phi_r = jnp.asarray(
        [jnp.matmul(jlt_mat[i], phi[i].T).T for i in range(query.shape[-2])]
    )
    top_mask = jnp.asarray(
        [jnp.matmul(phi_r[i], phi_r[i].T) for i in range(query.shape[-2])]
    )

    top_mask = jnp.pad(top_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)
    top_mask = jnp.pad(top_mask, [[0, 0], [0, 0], [1, 0]], constant_values=1.0)
    top_mask = top_mask[None, ...]

  elif top_type == 'grf':  #  masking with sparse graph random features
    m_matrix, taylor_params = (
        topological_params  #  M sparse matrix that stores locs
    )
    phis = [
        jnp.matmul(m_matrix[i], taylor_params[i])
        for i in range(query.shape[-2])
    ]
    top_mask = jnp.asarray(
        [jnp.matmul(phis[i], phis[i].T) for i in range(query.shape[-2])]
    )
    top_mask = jnp.pad(top_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)
    top_mask = jnp.pad(top_mask, [[0, 0], [0, 0], [1, 0]], constant_values=1.0)
    top_mask = top_mask[None, ...]

  elif top_type == 'unmasked':  #  unmasked attention
    top_mask = None

  else:
    raise ValueError('Topological masking type not recognised')

  if kernel_type == 'relu':
    if top_mask is not None:
      attn_weights = attn_weights * jnp.abs(top_mask) + 1e-8
    # normalize the attention weights
    attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)
  else:
    if top_mask is not None:
      attn_weights = attn_weights + top_mask
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
        keep_prob, dtype=dtype
    )
    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    topological_params: Optional[Array] = None,
    top_type: str = 'unmasked',
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype = jnp.float32,
    precision: Optional[lax.Precision] = None,
    nb_x_patches: int = 0,
    nb_y_patches: int = 0,
    kernel_type: str = 'relu',
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    topological_params: tensor defining parameters of topological masking.
    top_type: type of the topological masking used. 'toeplitz', 'heat, 'taylor'
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    nb_x_patches: number of patches in a fixed column.
    nb_y_patches: number of patches in a fixed row.
    kernel_type: type of the attention kernel used.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
      query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
      query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = topological_dot_product_attention_weights(
      query,
      key,
      topological_params,
      top_type,
      bias,
      mask,
      broadcast_dropout,
      dropout_rng,
      dropout_rate,
      deterministic,
      dtype,
      precision,
      nb_x_patches,
      nb_y_patches,
      kernel_type,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: float32)
    param_dtype: the dtype passed to parameter initializers (default: float32).
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly using
      dropout, whereas if true, the attention weights are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
    nb_x_patches: number of patches in a fixed column,
    nb_y_patches: number of patches in a fixed row,
    kernel_type: type of the attention kernel used.
  """

  num_heads: int
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[
      [
          Array,
          Array,
          Array,
          Optional[Any],
          Optional[str],
          Optional[Array],
          Optional[Array],
          bool,
          Optional[Any],
          float,
          bool,
          Any,
          Optional[Any],
          int,
          int,
      ],
      Array,
  ] = dot_product_attention
  decode: bool = False
  nb_x_patches: int = 0
  nb_y_patches: int = 0
  kernel_type: str = 'relu'
  top_type: str = 'unmasked'

  @compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      mask: Optional[Array] = None,
      deterministic: Optional[bool] = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape `[batch_sizes..., length, features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = merge_param('deterministic', self.deterministic,
                                  deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key', jnp.zeros, key.shape,
                                 key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(max_length) <= cur_index,
                tuple(batch_dims) + (1, 1, max_length),
            ),
        )

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # ~~~~~~INSTANTIATE VARS DEPENDING ON TOP TYPE. ~~~~~
    if self.top_type == 'toeplitz':  # block Toeplitz mask
      top_params = self.param(
          'toeplitz_params',
          ones,
          (
              query.shape[-2],
              4 * self.nb_x_patches * self.nb_y_patches,
          ),
      )

    elif self.top_type == 'heat':  # heat kernel with learnable lengthscale
      top_params = self.param('heat_params', ones, (query.shape[-2]))

    elif self.top_type == 'taylor':  # arbitrary Taylor expansion
      max_steps = 100  # maximum number of steps to use in taylor expansion
      decay_rate = 0.01  # decay rate at initialisation
      top_params = self.param(
          'taylor_params',
          decaying_init_func,
          query.shape[-2],
          max_steps,
          decay_rate,
      )

    elif self.top_type == 'learned_features':  # just learn top features
      max_steps = 10  # dimensionality of feature vectors
      top_params = self.param(
          'feature_params',
          ones,
          (query.shape[-2], self.nb_x_patches * self.nb_y_patches, max_steps),
      )

    elif self.top_type == 'jlt_taylor':   # learnable taylor exp with JLT
      max_steps = 100
      jlt_dim = 20
      adj_coefficients = self.param(
          'taylor_params', ones, (query.shape[-2], max_steps)
      )
      #  FOR A LEARNABLE JLT MATRIX:
      # jlt_mat = self.param(
      #    'jlt_mat',
      #    normal_initialiser,
      #    (query.shape[-2], jlt_dim, self.nb_x_patches * self.nb_y_patches),
      # )
      #  FOR A FIXED JLT MATRIX:
      jlt_rng = jax.random.key(np.random.randint(100000))  # horrible hack. TODO
      jlt_mat = self.variable(
          'batch_stats',
          'jlt_mat',
          normal_initialiser, jlt_rng,
          (query.shape[-2], jlt_dim, self.nb_x_patches * self.nb_y_patches),
      )
      # top_params = [jlt_mat, adj_coefficients]
      top_params = [jlt_mat.value, adj_coefficients]

    elif self.top_type == 'grf':  #  mask using sparse GRFs
      max_steps = 100
      p_halt = 0.1
      walks_per_vertex = 100
      decay_rate = 0.01
      #  variable for sparse matrix storing walker destinations
      m_matrix = self.variable(
          'batch_stats',
          'M',
          random_walk_utils.qmc_m_initialiser,  # or m_initialiser
          None,  # prng key. TODO
          query.shape[-2],
          self.nb_x_patches,
          self.nb_y_patches,
          walks_per_vertex,
          max_steps,
          p_halt,
          False,    #  antithetic?
          True,    #  repelling?
      )
      #  learmable Taylor expansion coefficients
      taylor_params = self.param(
          'taylor_params',
          decaying_init_func,
          query.shape[-2],
          max_steps,
          decay_rate,
      )
      top_params = [m_matrix.value, taylor_params]

    elif self.top_type == 'unmasked':  # no topological masking
      top_params = None

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        top_params,
        top_type=self.top_type,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision,
        nb_x_patches=self.nb_x_patches,
        nb_y_patches=self.nb_y_patches,
        kernel_type=self.kernel_type)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name='out')(
            x)
    return out


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @compact
  def __call__(self,
               inputs_q: Array,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    return super().__call__(
        inputs_q, inputs_q, mask, deterministic=deterministic)


# mask-making utility functions


def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable[..., Any] = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: Dtype = jnp.float32):
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
  attention weights will be `[batch..., heads, len_q, len_kv]` and this
  function will produce `[batch..., 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  mask = pairwise_fn(
      jnp.expand_dims(query_input, axis=-1),
      jnp.expand_dims(key_input, axis=-2))
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: Dtype = jnp.float32):
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
  will be `[batch..., heads, len, len]` and this function will produce a
  causal mask of shape `[batch..., 1, len, len]`.

  Args:
    x: input array of shape `[batch..., len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: Dtype = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: dtype for the returned mask.

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)
