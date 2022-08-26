"""Loss functions."""

from typing import Optional

from jax import nn
import jax.numpy as jnp


def _rows_to_columns_nce_loss(
    scores: jnp.ndarray,  # Shape: (N, N)
    where: Optional[jnp.ndarray] = None,  # Shape: broadcastable with `scores`
    initial: Optional[float] = None,
) -> jnp.ndarray:  # Shape: (N,)
  """Computes the InfoNCE loss from rows to columns."""
  return -nn.log_softmax(scores, where=where, initial=initial).diagonal()


def nce_loss(
    scores: jnp.ndarray,  # Shape: (N, N)
    where: Optional[jnp.ndarray] = None,  # Shape: broadcastable with (N,)
    initial: Optional[float] = None,
) -> jnp.ndarray:  # Shape: (N,)
  return (_rows_to_columns_nce_loss(scores, where=where, initial=initial) +
          _rows_to_columns_nce_loss(scores.T, where=where, initial=initial))
