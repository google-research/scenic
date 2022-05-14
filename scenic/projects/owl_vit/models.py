"""Implementation of Conditional ViTPlus detection model.

The implementation allows for: 1) using label-embeddings to use as fixed class
projection, 2) (optionally) conditioning the decoder on a set of given labels.
"""

from typing import Any, Dict, List, Mapping, Optional

import flax.linen as nn
from flax.training import checkpoints
import jax.numpy as jnp
import ml_collections
from scenic.projects.owl_vit import layers
from scenic.projects.owl_vit import utils
from scenic.projects.owl_vit.clip import model as clip_model
from scenic.projects.owl_vit.clip import tokenizer as clip_tokenizer


class TextZeroShotDetectionModule(nn.Module):
  """Text-query-based ViT+ model with detection head.

  This module computes joint text and image embeddings which are then
  used for localized prediction of bboxes and classes.

  Attributes:
    body_configs: Configurations of the image-text module.
    normalize: Whether to normalize the output of the model and the
      label_embeddings before computing the class logits.
    box_bias: Type of box bias - one of 'location', 'size' or 'both'.
    mask_size: The height (and width) of masks predicted by the mask head. If
      None, no mask prediction will occur.
  """

  body_configs: ml_collections.ConfigDict
  normalize: bool = False
  box_bias: str = 'both'
  mask_size: Optional[int] = None

  def tokenize(self, text: str, max_token_len: int = 16) -> List[int]:
    return clip_tokenizer.tokenize(text, max_token_len)

  @nn.nowrap
  def load_variables(self, checkpoint_path: str) -> Mapping[str, Any]:
    restored = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    return {'params': restored['optimizer']['target']}

  def setup(self):
    self._embedder = layers.ImageTextEmbedder(
        self.body_configs, name='backbone')
    if 'out_dim' in self.body_configs:
      out_dim = self.body_configs.out_dim
    elif 'type' in self.body_configs and self.body_configs.type == 'clip':
      out_dim = clip_model.CONFIGS[self.body_configs.variant]['embed_dim']
    else:
      # Attempt to lazily determine the output dimension.
      out_dim = None
    self._class_head = layers.ClassPredictor(
        out_dim=out_dim,
        normalize=self.normalize, name='class_head')
    self._box_head = layers.PredictorMLP(
        mlp_dim=None, out_dim=4, num_layers=3,
        out_activation=None, name='obj_box_head')
    if self.mask_size is not None:
      self._mask_head = layers.PredictorMLP(
          mlp_dim=None,
          out_dim=self.mask_size * self.mask_size,
          num_layers=3,
          out_activation=None,
          name='obj_mask_head')

  def box_predictor(self, image_features: jnp.ndarray,
                    feature_map: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Computes predicted bounding boxes.

    Args:
      image_features: Features extracted from the image, returned by the
        `embedder` function.
      feature_map: A spatial re-arrangement of image_features, also returned by
        the `embedder` function.

    Returns:
      list of predicted boxes (cxcywh normalized to 0, 1) nested within
        a dictionary.
    """
    # Bounding box detection head [b, num_patches, 4].
    pred_boxes = self._box_head(image_features)
    # We compute the location of each token on the grid and use it to compute
    # a bias for the bbox prediction, i.e., each token is biased towards
    # predicting its location on the grid as the center.
    pred_boxes += utils.compute_box_bias(feature_map, kind=self.box_bias)
    pred_boxes = nn.sigmoid(pred_boxes)
    return {'pred_boxes': pred_boxes}

  def class_predictor(
      self,
      image_features: jnp.ndarray,
      query_embeddings: Optional[jnp.ndarray] = None,
      query_mask: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    """Applies the class head to the image features.

    Args:
      image_features: Features extracted from the image embedder.
      query_embeddings: Optional list of (or image) embeddings. If no embeddings
        are provided, no logits will be computed and only the class embeddings
        for the image will be returned.
      query_mask: Must be provided with query_embeddings. A mask indicating
        which query embeddings are valid.

    Returns:
      A dictionary containing the class_embeddings and the pred_logits if
        query_embeddings and query_mask are provided.
    """
    return self._class_head(image_features, query_embeddings, query_mask)

  def mask_predictor(self, image_features) -> Dict[str, jnp.ndarray]:
    """Predicts (cropped) segmentation masks from the image features.

    Args:
      image_features: Features extracted from the image embedder.

    Returns:
      A dictionary containing the predicted segmentation masks. The mask at
        index i corresponds to the predicted box in `pred_boxes` at index i.
    """
    if self.mask_size is None:
      raise ValueError('Must pass mask_size to use mask head.')
    pred_masks = self._mask_head(image_features)
    batch_size = image_features.shape[0]
    return {
        'pred_masks':
            jnp.reshape(pred_masks,
                        (batch_size, -1, self.mask_size, self.mask_size))
    }

  def image_embedder(self, images: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Embeds images into feature maps.

    Args:
      images: images of shape (batch, self.input_size, self.input_size, 3).
        Images should be in range [-1., 1.] with padding set to 0 and at the
        bottom right of the image.
      train: Whether or not we are in training mode.

    Returns:
      A 2D map of image features.
    """
    image_features, _ = self._embedder(images=images, train=train)
    return utils.seq2img(images, image_features)

  def text_embedder(self, text_queries: jnp.ndarray,
                    train: bool) -> jnp.ndarray:
    """Embeds text into features.

    Args:
      text_queries: jnp.int32 tokenized text queries of shape [..., num_tokens].
      train: Whether or not we are in training mode.

    Returns:
      An array of the same shape as text_queries, except for the last dimension,
      which is num_dimensions instead of num_tokens.
    """
    _, text_features = self._embedder(texts=text_queries, train=train)
    return text_features

  def __call__(self,
               inputs: jnp.ndarray,
               text_queries: jnp.ndarray,
               train: bool,
               *,
               debug: bool = False) -> Mapping[str, Any]:
    """Applies TextZeroShotDetectionModule on the input.

    Args:
      inputs: Images [batch_size, height, width, 3].
      text_queries: Queries to condition the model on. Queries starting with 0
        stand for padding [batch_size=b, num_queries=q, max_query_length=l].
      train: Whether it is training.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback. Not used.

    Returns:
      Outputs dict with items:
        pred_logits: Class logits [b, num_patches, num_queries + 1].
        pred_boxes: Predicted bounding boxes [b, num_patches, 4].
        feature_map: Image embeddings 2d feature map [b, sp, sp, img_emb_dim].
    """
    del debug
    # Embed images:
    feature_map = self.image_embedder(inputs, train)
    b, h, w, d = feature_map.shape
    image_features = jnp.reshape(feature_map, (b, h * w, d))

    # Embed queries:
    query_embeddings = self.text_embedder(text_queries, train)
    # If first token is 0, then this is a padded query [b, q].
    query_mask = (text_queries[..., 0] > 0).astype(jnp.float32)

    outputs = {
        'feature_map': feature_map,
        'query_embeddings': query_embeddings,
    }

    # Classification [b, num_patches, num_queries+1]:
    outputs.update(
        self.class_predictor(image_features, query_embeddings, query_mask))

    # Predict boxes:
    outputs.update(self.box_predictor(image_features, feature_map))

    # Predict masks:
    if self.mask_size is not None:
      outputs.update(self.mask_predictor(image_features))

    return outputs
