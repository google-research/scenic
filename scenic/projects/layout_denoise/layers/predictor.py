"""Output layers."""
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from scenic.projects.layout_denoise.layers import common


class ObjectClassPredictor(nn.Module):
  """Linear Projection block for predicting classification."""
  num_classes: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: jnp.float32 = .0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray,
               deterministic: bool = True) -> jnp.ndarray:
    """Applies Linear Projection to inputs.

    Args:
      inputs: Input data.
      deterministic: Whether to use dropout.
    Returns:
      Output of Linear Projection block.
    """
    inputs = nn.Dropout(rate=self.dropout_rate)(
        inputs, deterministic=deterministic)
    bias_range = 1. / np.sqrt(inputs.shape[-1])
    return nn.Dense(
        self.num_classes,
        kernel_init=common.pytorch_kernel_init(dtype=self.dtype),
        bias_init=common.uniform_initializer(
            -bias_range, bias_range, self.dtype),
        dtype=self.dtype)(
            inputs)

