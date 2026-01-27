"""Memory-efficient attention mechanisms for Scenic.

This module implements efficient attention variants that reduce memory consumption
and computational complexity compared to standard attention:

1. FlashAttention: Memory-efficient attention using block-wise computation
2. LinearAttention: O(N) complexity attention using kernel approximations

These implementations are compatible with Flax's nn.Module API and can be used
as drop-in replacements for standard MultiHeadDotProductAttention.

Reference:
  FlashAttention: Dao et al. "FlashAttention: Fast and Memory-Efficient Exact
                  Attention with IO-Awareness" (2022)
  Linear Attention: Katharopoulos et al. "Transformers are RNNs: Fast
                    Autoregressive Transformers with Linear Attention" (2020)
"""

from typing import Any, Callable, Optional, Tuple
import functools

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax


Array = jnp.ndarray
PRNGKey = jax.Arraygit
Shape = Tuple[int, ...]
Dtype = Any


def flash_attention_kernel(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: Optional[lax.Precision] = None,
    block_size: int = 512,
) -> Array:
  """Flash attention kernel with tiled computation.

  Implements block-wise attention computation to reduce memory usage.
  Uses online softmax to avoid materializing full attention matrix.

  Note: Dropout is only supported when both sequence lengths are <= block_size.
  For longer sequences using the blocked path, dropout is not applied.

  Args:
    query: Query array [batch, q_length, num_heads, qk_depth_per_head]
    key: Key array [batch, kv_length, num_heads, qk_depth_per_head]
    value: Value array [batch, kv_length, num_heads, v_depth_per_head]
    bias: Optional bias to add to attention logits
    mask: Optional mask [batch, num_heads, q_length, kv_length]
    dropout_rng: Optional PRNG key for dropout
    dropout_rate: Dropout rate
    deterministic: If true, no dropout is applied
    dtype: Output dtype
    precision: Numerical precision (unused, kept for API compatibility)
    block_size: Size of blocks for tiled computation

  Returns:
    Attention output [batch, q_length, num_heads, v_depth_per_head]
  """
  del precision  # Unused, kept for API compatibility

  batch_size, q_len, num_heads, head_dim = query.shape
  kv_len = key.shape[1]

  scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

  # For small sequences, use standard attention
  if q_len <= block_size and kv_len <= block_size:
    attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn_weights = attn_weights * scale

    if bias is not None:
      attn_weights = attn_weights + bias
    if mask is not None:
      big_neg = jnp.finfo(attn_weights.dtype).min
      attn_weights = jnp.where(mask, attn_weights, big_neg)

    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    if not deterministic and dropout_rate > 0.0:
      keep_prob = 1.0 - dropout_rate
      dropout_shape = list(attn_weights.shape)
      keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      attn_weights = jnp.where(keep, attn_weights / keep_prob, 0)

    output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
    if dtype is not None:
      output = output.astype(dtype)
    return output

  # Blocked computation for longer sequences
  num_q_blocks = (q_len + block_size - 1) // block_size
  num_kv_blocks = (kv_len + block_size - 1) // block_size

  # Initialize with shape (batch, q_len, num_heads, ...) for consistency
  output = jnp.zeros((batch_size, q_len, num_heads, head_dim), dtype=query.dtype)
  # Statistics in (batch, q_len, num_heads) to match output layout
  max_scores = jnp.full((batch_size, q_len, num_heads), -jnp.inf)
  sum_exp = jnp.zeros((batch_size, q_len, num_heads))

  for kv_block_idx in range(num_kv_blocks):
    kv_start = kv_block_idx * block_size
    kv_end = min(kv_start + block_size, kv_len)

    key_block = key[:, kv_start:kv_end]
    value_block = value[:, kv_start:kv_end]

    for q_block_idx in range(num_q_blocks):
      q_start = q_block_idx * block_size
      q_end = min(q_start + block_size, q_len)

      query_block = query[:, q_start:q_end]

      # Compute attention scores: (batch, num_heads, q_block, k_block)
      scores = jnp.einsum('...qhd,...khd->...hqk', query_block, key_block)
      scores = scores * scale

      if bias is not None:
        bias_block = bias[:, :, q_start:q_end, kv_start:kv_end]
        scores = scores + bias_block
      if mask is not None:
        mask_block = mask[:, :, q_start:q_end, kv_start:kv_end]
        big_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask_block, scores, big_neg)

      # Online softmax: block_max has shape (batch, num_heads, q_block)
      block_max = jnp.max(scores, axis=-1)
      # Transpose to (batch, q_block, num_heads) to match our statistics layout
      block_max_t = jnp.transpose(block_max, (0, 2, 1))

      old_max = max_scores[:, q_start:q_end, :]
      new_max = jnp.maximum(old_max, block_max_t)

      # Rescale factors
      exp_old = jnp.exp(old_max - new_max)  # (batch, q_block, num_heads)
      # For scores, need (batch, num_heads, q_block, 1)
      new_max_for_scores = jnp.transpose(new_max, (0, 2, 1))[..., None]
      exp_scores = jnp.exp(scores - new_max_for_scores)

      # Update sum: sum over k dimension, transpose result
      exp_scores_sum = jnp.sum(exp_scores, axis=-1)  # (batch, num_heads, q_block)
      exp_scores_sum_t = jnp.transpose(exp_scores_sum, (0, 2, 1))

      old_sum = sum_exp[:, q_start:q_end, :]
      new_sum = exp_old * old_sum + exp_scores_sum_t

      # Update output
      output_block = output[:, q_start:q_end, :, :]
      # Weighted values from current block: (batch, q_block, num_heads, head_dim)
      weighted_values = jnp.einsum('...hqk,...khd->...qhd', exp_scores, value_block)
      output_block = output_block * exp_old[..., None] + weighted_values

      # Store updates
      output = output.at[:, q_start:q_end, :, :].set(output_block)
      max_scores = max_scores.at[:, q_start:q_end, :].set(new_max)
      sum_exp = sum_exp.at[:, q_start:q_end, :].set(new_sum)

  # Final normalization
  output = output / sum_exp[..., None]

  if dtype is not None:
    output = output.astype(dtype)

  return output


def linear_attention_kernel(
    query: Array,
    key: Array,
    value: Array,
    feature_map: Optional[Callable] = None,
    eps: float = 1e-6,
    dtype: Optional[Dtype] = None,
) -> Array:
  """Linear attention kernel with O(N) complexity.

  Uses kernel approximation to achieve linear complexity in sequence length.
  Based on "Transformers are RNNs" paper.

  Args:
    query: Query array [batch, q_length, num_heads, qk_depth_per_head]
    key: Key array [batch, kv_length, num_heads, qk_depth_per_head]
    value: Value array [batch, kv_length, num_heads, v_depth_per_head]
    feature_map: Optional feature map function (default: elu+1)
    eps: Small constant for numerical stability
    dtype: Output dtype

  Returns:
    Attention output [batch, q_length, num_heads, v_depth_per_head]
  """
  if feature_map is None:
    # Default feature map: elu(x) + 1
    feature_map = lambda x: jax.nn.elu(x) + 1.0

  # Apply feature map
  query = feature_map(query)
  key = feature_map(key)

  # Linear attention: O(N) complexity
  # Compute K^T V first (reduces to O(d^2) instead of O(N^2))
  kv = jnp.einsum('...khd,...khm->...hdm', key, value)

  # Then compute Q(K^T V)
  output = jnp.einsum('...qhd,...hdm->...qhm', query, kv)

  # Normalization
  normalizer = jnp.einsum('...qhd,...khd->...qh', query, key)
  normalizer = jnp.maximum(normalizer, eps)
  output = output / normalizer[..., None]

  if dtype is not None:
    output = output.astype(dtype)

  return output


class FlashAttention(nn.Module):
  """Flash Attention layer for memory-efficient attention computation.

  This implements the FlashAttention algorithm which reduces memory usage
  from O(N²) to O(N) through block-wise computation and online softmax.

  Note: Dropout is only supported when both sequence lengths are <= block_size.
  For longer sequences, dropout is not applied regardless of dropout_rate.

  Attributes:
    num_heads: Number of attention heads
    dtype: Dtype of computation (default: inferred from inputs)
    param_dtype: Dtype of parameters
    qkv_features: Dimension of query, key, and value features
    out_features: Dimension of output features
    broadcast_dropout: Whether to broadcast dropout mask
    dropout_rate: Dropout rate
    deterministic: If true, no dropout
    precision: Numerical precision
    kernel_init: Initializer for projection kernels
    bias_init: Initializer for biases
    use_bias: Whether to use bias
    block_size: Block size for tiled computation (default: 512)

  Example:
    >>> import flax.linen as nn
    >>> import jax.numpy as jnp
    >>>
    >>> # Create layer
    >>> attn = FlashAttention(num_heads=8, qkv_features=512)
    >>>
    >>> # Initialize
    >>> x = jnp.ones((2, 100, 512))  # [batch, seq_len, features]
    >>> variables = attn.init(jax.random.PRNGKey(0), x)
    >>>
    >>> # Apply
    >>> output = attn.apply(variables, x)
    >>> print(output.shape)  # (2, 100, 512)
  """

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: Optional[lax.Precision] = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  use_bias: bool = True
  block_size: int = 512

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_k: Optional[Array] = None,
      inputs_v: Optional[Array] = None,
      mask: Optional[Array] = None,
      deterministic: Optional[bool] = None,
  ) -> Array:
    """Apply flash attention.

    Args:
      inputs_q: Query input [batch, q_length, features]
      inputs_k: Key input [batch, kv_length, features] (default: inputs_q)
      inputs_v: Value input [batch, kv_length, features] (default: inputs_k)
      mask: Attention mask [batch, num_heads, q_length, kv_length]
      deterministic: If true, no dropout

    Returns:
      Output array [batch, q_length, out_features]
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        f'qkv_features ({qkv_features}) must be divisible by '
        f'num_heads ({self.num_heads})')

    head_dim = qkv_features // self.num_heads

    # Set default values for k and v
    if inputs_k is None:
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    # Dense layers for Q, K, V projections
    dense = functools.partial(
        nn.Dense,
        features=qkv_features,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
    )

    query = dense(name='query')(inputs_q)
    key = dense(name='key')(inputs_k)
    value = dense(name='value')(inputs_v)

    # Reshape to [batch, length, num_heads, head_dim]
    batch_size = query.shape[0]
    query = query.reshape(batch_size, -1, self.num_heads, head_dim)
    key = key.reshape(batch_size, -1, self.num_heads, head_dim)
    value = value.reshape(batch_size, -1, self.num_heads, head_dim)

    # Determine dropout behavior
    if deterministic is None:
      deterministic = self.deterministic
    if deterministic is None:
      deterministic = True

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # Apply flash attention kernel
    x = flash_attention_kernel(
        query=query,
        key=key,
        value=value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision,
        block_size=self.block_size,
    )

    # Reshape and project output
    x = x.reshape(batch_size, -1, qkv_features)

    out = nn.Dense(
        features=features,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        name='out',
    )(x)

    return out


class LinearAttention(nn.Module):
  """Linear Attention layer with O(N) complexity.

  This implements linear attention using kernel approximations, reducing
  complexity from O(N²) to O(N) for sequence length N.

  Attributes:
    num_heads: Number of attention heads
    dtype: Dtype of computation
    param_dtype: Dtype of parameters
    qkv_features: Dimension of query, key, and value features
    out_features: Dimension of output features
    precision: Numerical precision
    kernel_init: Initializer for projection kernels
    bias_init: Initializer for biases
    use_bias: Whether to use bias
    feature_map: Feature map function (default: elu+1)
    eps: Small constant for numerical stability

  Example:
    >>> import flax.linen as nn
    >>> import jax.numpy as jnp
    >>>
    >>> # Create layer
    >>> attn = LinearAttention(num_heads=8, qkv_features=512)
    >>>
    >>> # Initialize and apply
    >>> x = jnp.ones((2, 1000, 512))  # Can handle long sequences efficiently
    >>> variables = attn.init(jax.random.PRNGKey(0), x)
    >>> output = attn.apply(variables, x)
    >>> print(output.shape)  # (2, 1000, 512)
  """

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  precision: Optional[lax.Precision] = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  use_bias: bool = True
  feature_map: Optional[Callable] = None
  eps: float = 1e-6

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_k: Optional[Array] = None,
      inputs_v: Optional[Array] = None,
  ) -> Array:
    """Apply linear attention.

    Args:
      inputs_q: Query input [batch, q_length, features]
      inputs_k: Key input [batch, kv_length, features] (default: inputs_q)
      inputs_v: Value input [batch, kv_length, features] (default: inputs_k)

    Returns:
      Output array [batch, q_length, out_features]
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        f'qkv_features ({qkv_features}) must be divisible by '
        f'num_heads ({self.num_heads})')

    head_dim = qkv_features // self.num_heads

    # Set default values for k and v
    if inputs_k is None:
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    # Dense layers for Q, K, V projections
    dense = functools.partial(
        nn.Dense,
        features=qkv_features,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
    )

    query = dense(name='query')(inputs_q)
    key = dense(name='key')(inputs_k)
    value = dense(name='value')(inputs_v)

    # Reshape to [batch, length, num_heads, head_dim]
    batch_size = query.shape[0]
    query = query.reshape(batch_size, -1, self.num_heads, head_dim)
    key = key.reshape(batch_size, -1, self.num_heads, head_dim)
    value = value.reshape(batch_size, -1, self.num_heads, head_dim)

    # Apply linear attention kernel
    x = linear_attention_kernel(
        query=query,
        key=key,
        value=value,
        feature_map=self.feature_map,
        eps=self.eps,
        dtype=self.dtype,
    )

    # Reshape and project output
    x = x.reshape(batch_size, -1, qkv_features)

    out = nn.Dense(
        features=features,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        name='out',
    )(x)

    return out