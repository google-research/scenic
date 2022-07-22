"""Modules and functions used for zero-shot model."""

from typing import Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.owl_vit.clip import layers as clip_layers
from scenic.projects.owl_vit.clip import model as clip_model


class PredictorMLP(nn.Module):
  """FFN block for predicting bounding box coordinates.

  Attributes:
    out_dim: Size of output of this mlp.
    num_layers: Number of layers.
    mlp_dim: Size of hidden dimension of dense layers.
    hidden_activation: Activation function of hidden layers.
    out_activation: Activation of the output.
    dtype: Data type, e.g. jnp.float32.
  """
  out_dim: int
  num_layers: int = 1
  mlp_dim: Optional[int] = None
  hidden_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = nn.gelu
  out_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies FFN MLP block to inputs for prediction."""
    x = inputs
    mlp_dim = self.mlp_dim or x.shape[-1]
    for _ in range(self.num_layers-1):
      x = nn.Dense(mlp_dim, dtype=self.dtype)(x)
      if self.hidden_activation is not None:
        x = self.hidden_activation(x)

    x = nn.Dense(self.out_dim, kernel_init=nn.zeros)(x)
    if self.out_activation is not None:
      x = self.out_activation(x)  # pylint: disable=not-callable
    return x


class ClassPredictor(nn.Module):
  """Zero-shot instance class predictor."""
  normalize: bool = False
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      query_embeddings: Optional[jnp.ndarray] = None,
      query_mask: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
    """Computes class prediction logits.

    Args:
      x: Image features [batch_size, num_patches, emb_dim].
      query_embeddings: The embeddings to classify against of shape [batch_size,
        num_queries, out_dim]. If not specified, only the image class embeddings
        will be returned.
      query_mask: Mask indicating whether query is real (1) or padding (0), of
        shape [batch_size, num_queries].
    Returns:
      Dict with keys 'class_embeddings' and, if query embeddings were provided,
      'pred_logits'.
    """
    if self.out_dim is not None:
      out_dim = self.out_dim
    elif query_embeddings is not None:
      out_dim = query_embeddings.shape[-1]
    else:
      raise ValueError('Unable to infer class head shape. Please pass out_dim.')

    image_class_emb = nn.Dense(
        out_dim, kernel_init=nn.initializers.normal(1e-6))(x)
    if query_embeddings is None:
      return {'class_embeddings': image_class_emb}
    assert out_dim == query_embeddings.shape[-1]

    if self.normalize:
      image_class_emb /= jnp.linalg.norm(
          image_class_emb, axis=-1, keepdims=True) + 1e-6
      query_embeddings /= jnp.linalg.norm(
          query_embeddings, axis=-1, keepdims=True) + 1e-6

    assert query_embeddings.ndim > 2, ('Expects shape (batch, query, out_dim). '
                                       f'Got {query_embeddings.shape}')
    pred_logits = jnp.einsum(
        '...pd,...qd->...pq', image_class_emb, query_embeddings)

    # Apply a learnable shift and scale to logits:
    logit_shift = nn.Dense(1, name='logit_shift')(x)
    logit_scale = nn.Dense(1, use_bias=True, name='logit_scale')(x)
    logit_scale = nn.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    if query_mask is not None:
      if query_mask.ndim > 1:
        query_mask = jnp.expand_dims(query_mask, axis=-2)
      pred_logits = jnp.where(query_mask == 0, -1e6, pred_logits)

    return {'pred_logits': pred_logits, 'class_embeddings': image_class_emb}


class ImageTextEmbedder(nn.Module):
  """Embeds images and texts using selected backbone."""
  embed_configs: ml_collections.ConfigDict

  @nn.compact
  def __call__(
      self,
      *,
      images: Optional[jnp.ndarray] = None,
      texts: Optional[jnp.ndarray] = None,
      train: bool = False
  ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Embeds text using selected backbone and configuration."""
    texts_shape = None
    if texts is not None:
      texts_shape = texts.shape
      if len(texts_shape) > 2:
        texts = texts.reshape(-1, texts_shape[-1])

    emb_type = self.embed_configs.get('type')
    if emb_type == 'clip':
      model_config = clip_model.CONFIGS[self.embed_configs['variant']]
      model_config['vision_return_map'] = True
      # Copy over additional CLIP config settings.
      for name in [
          'text_stochastic_droplayer_rate', 'vision_stochastic_droplayer_rate']:
        if self.embed_configs.get(name) is not None:
          model_config[name] = self.embed_configs[name]
      model = clip_layers.CLIP(**model_config, name='clip')
      # Input images should have range (0.0, 1.0). Shift them to CLIP range:
      if images is not None:
        images = clip_model.normalize_image(images)
      # Don't normalize image and text embeddings:
      img_emb, txt_emb = model(
          images, texts, normalize=False, deterministic=not train)
      # Drop or merge class embedding token.
      # TODO(mnn): Remove after the preferred class token merging scheme is
      # determined.
      if img_emb is not None:
        merge_class_token = self.embed_configs.get('merge_class_token', 'sum')
        if merge_class_token == 'drop':
          img_emb = img_emb[:, 1:, :]   # [B, P, emb_dim]
        else:
          class_token_out = jnp.broadcast_to(
              img_emb[:, :1, :],
              np.array(img_emb.shape) - (0, 1, 0))
          if merge_class_token == 'sum':
            img_emb = img_emb[:, 1:, :] + class_token_out   # [B, P, emb_dim]
          elif merge_class_token == 'mul':
            img_emb = img_emb[:, 1:, :] * class_token_out   # [B, P, emb_dim]
          elif merge_class_token == 'sum-ln':
            img_emb = img_emb[:, 1:, :] + class_token_out   # [B, P, emb_dim]
            img_emb = nn.LayerNorm(name='merged_class_token')(img_emb)
          elif merge_class_token == 'mul-ln':
            img_emb = img_emb[:, 1:, :] * class_token_out   # [B, P, emb_dim]
            img_emb = nn.LayerNorm(name='merged_class_token')(img_emb)
    else:
      raise NotImplementedError(f'Unknown embedding type {emb_type}.')

    if txt_emb is not None and len(texts_shape) > 2:
      txt_emb = txt_emb.reshape(texts_shape[:-1] + (-1,))
    return img_emb, txt_emb
