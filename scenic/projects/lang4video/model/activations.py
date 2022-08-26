"""Activation functions."""

import flax.linen as nn
import jax.numpy as jnp


def mish(x: jnp.ndarray) -> jnp.ndarray:
  return x * nn.tanh(nn.softplus(x))


ACTIVATIONS_BY_NAME = {
    'gelu': nn.gelu,
    'mish': mish,
    'relu': nn.relu,
    'swish': nn.swish,  # Same as silu.
    'tanh': nn.tanh,
}
