"""Customized Mlp block for MatViT.
"""
from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.model_lib.layers import nn_layers

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      deterministic: bool,
      matvit_mask: Optional[Any] = None,
  ):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(inputs)

    if matvit_mask is not None:
      x = nn_layers.IdentityLayer(name='mlp1')(
          self.activation_fn(x * matvit_mask)
      )
    else:
      x = nn_layers.IdentityLayer(name='mlp1')(self.activation_fn(x))

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
    )(x)
    output = nn_layers.IdentityLayer(name='mlp2')(output)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic
    )
    return output
