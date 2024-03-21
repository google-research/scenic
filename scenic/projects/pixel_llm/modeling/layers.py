"""Layer utilis."""

import flax.linen as nn


class LinearProjectLayers(nn.Module):
  """Linear projection layer."""
  emb_dim: int = 1024
  use_projection_ln: bool = True

  @nn.compact
  def __call__(self, x, train=False):
    # The name `visual_projection.x` is for a historical reason to load
    # weights for other decoders. This is not meaningful here now.
    x = nn.Dense(
        self.emb_dim, name='visual_projection.0',
        kernel_init=nn.initializers.normal(stddev=0.02))(
            x)  # (batch_size, feature_length, hidden_size)
    if self.use_projection_ln:
      x = nn.LayerNorm(
          epsilon=1e-5, name='visual_projection.1')(x)
    return x
