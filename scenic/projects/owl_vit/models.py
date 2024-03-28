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

"""Implementation of the OWL-ViT detection model."""

import copy
from typing import Any, Dict, List, Mapping, Optional

import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.owl_vit import layers
from scenic.projects.owl_vit import matching_base_models
from scenic.projects.owl_vit import utils
from scenic.projects.owl_vit.clip import model as clip_model
from scenic.projects.owl_vit.clip import tokenizer as clip_tokenizer


Params = layers.Params


def _fix_old_layernorm(transformer_params):
  """Fix layer norm numbering of old checkpoints."""
  if (
      'resblocks.0' in transformer_params
      and 'ln_0' in transformer_params['resblocks.0']
  ):
    # This checkpoint has the new format.
    return transformer_params

  fixed_params = copy.deepcopy(transformer_params)
  for resblock in fixed_params.values():
    resblock['ln_0'] = resblock.pop('ln_1')
    resblock['ln_1'] = resblock.pop('ln_2')

  return fixed_params


def _fix_resblock_naming(transformer_params):
  """Fix resblock naming of old checkpoints."""
  if 'resblocks_0' in transformer_params:
    # This checkpoint is already converted.
    return transformer_params

  fixed_params = copy.deepcopy(transformer_params)
  old_keys = list(fixed_params.keys())
  for old_key in old_keys:
    new_key = old_key.replace('.', '_')
    fixed_params[new_key] = fixed_params.pop(old_key)

  return fixed_params


def _fix_old_checkpoints(params):
  """Makes old checkpoints forward-compatible."""
  if 'clip' in params['backbone']:
    # Fix the layer norm indexing.
    params['backbone']['clip']['visual']['transformer'] = _fix_old_layernorm(
        params['backbone']['clip']['visual']['transformer']
    )
    params['backbone']['clip']['text']['transformer'] = _fix_old_layernorm(
        params['backbone']['clip']['text']['transformer']
    )

    # Fix the resblock naming.
    params['backbone']['clip']['visual']['transformer'] = _fix_resblock_naming(
        params['backbone']['clip']['visual']['transformer']
    )
    params['backbone']['clip']['text']['transformer'] = _fix_resblock_naming(
        params['backbone']['clip']['text']['transformer']
    )
  return params


class TextZeroShotDetectionModule(nn.Module):
  """Text-query-based OWL-ViT model.

  This module computes joint text and image embeddings which are then
  used for localized prediction of bounding boxes and classes.

  Attributes:
    body_configs: Configurations of the image-text module.
    objectness_head_configs: Configurations for the (optional) objectness head.
    mask_head_configs: Configurations for the (optional) mask head.
    normalize: Whether to normalize the output of the model and the
      label_embeddings before computing the class logits.
    box_bias: Type of box bias - one of 'location', 'size' or 'both'.
  """

  body_configs: ml_collections.ConfigDict
  objectness_head_configs: Optional[ml_collections.ConfigDict] = None
  mask_head_configs: Optional[ml_collections.ConfigDict] = None
  normalize: bool = False
  box_bias: str = 'both'

  def tokenize(self, text: str, max_token_len: int = 16) -> List[int]:
    return clip_tokenizer.tokenize(text, max_token_len)

  @nn.nowrap
  def load_variables(self, checkpoint_path: str) -> Mapping[str, Any]:
    restored = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    if 'optimizer' in restored:
      # Pre-Optax checkpoint:
      params = restored['optimizer']['target']
    else:
      params = restored['params']
    params = _fix_old_checkpoints(params)
    return {'params': params}

  def setup(self):

    self._embedder = layers.ClipImageTextEmbedder(
        self.body_configs, name='backbone')

    if self.objectness_head_configs is not None:
      self._objectness_head = layers.PredictorMLP(
          mlp_dim=None, out_dim=1, num_layers=3,
          out_activation=None, name='objectness_head')

    self._class_head = layers.ClassPredictor(
        out_dim=clip_model.CONFIGS[self.body_configs.variant]['embed_dim'],
        normalize=self.normalize, name='class_head')

    self._box_head = layers.PredictorMLP(
        mlp_dim=None, out_dim=4, num_layers=3,
        out_activation=None, name='obj_box_head')

    if self.mask_head_configs is not None:
      self._mask_head = layers.BoxMaskHead(
          **self.mask_head_configs,  # pylint: disable=not-a-mapping
          name='obj_mask_head')

  def objectness_predictor(
      self, image_features: jnp.ndarray, train: bool = False
  ) -> Dict[str, jnp.ndarray]:
    """Predicts the probability that each image feature token is an object.

    Args:
      image_features: Features extracted from the image.
      train: Whether or not we are in training mode.

    Returns:
      Objectness scores, in a dictionary.
    """
    del train
    # TODO(b/215588365): Need local variable to work around pytype bug.
    objectness_head_configs = self.objectness_head_configs
    if objectness_head_configs is None:
      raise ValueError('Must pass objectness_configs to use objectness head.')
    if objectness_head_configs.stop_gradient:
      image_features = jax.lax.stop_gradient(image_features)
    objectness_logits = self._objectness_head(image_features)
    return {'objectness_logits': objectness_logits[..., 0]}

  def box_predictor(
      self,
      *,
      image_features: jnp.ndarray,
      feature_map: jnp.ndarray,
      keep_image_tokens: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
    """Predicts bounding boxes from image features.

    Args:
      image_features: Features extracted from the image, flattened into a 1d
        sequence of tokens.
      feature_map: A 2d spatial re-arrangement of image_features.
      keep_image_tokens: If keep_image_tokens is not None, this indicates that
        image_features is a subset of tokens of the full grid. keep_image_tokens
        then contains the 1d indices of the kept tokens within the full token
        sequence. In that case, feature_map will contain dummy values at the
        dropped locations.

    Returns:
      List of predicted boxes (cxcywh normalized to 0, 1) nested within
        a dictionary.
    """
    # Bounding box detection head [b, num_patches, 4].
    pred_boxes = self._box_head(image_features)

    # We compute the location of each token on the grid and use it to compute
    # a bias for the bbox prediction, i.e., each token is biased towards
    # predicting its location on the grid as the center.
    box_bias = utils.compute_box_bias(
        feature_map=feature_map, kind=self.box_bias
    )

    if keep_image_tokens is not None:
      box_bias = jnp.take_along_axis(
          box_bias[None, ...], keep_image_tokens[..., None], axis=-2
      )

    pred_boxes += box_bias
    pred_boxes = nn.sigmoid(pred_boxes)
    return {'pred_boxes': pred_boxes}

  def class_predictor(
      self,
      image_features: jnp.ndarray,
      query_embeddings: Optional[jnp.ndarray] = None,
      query_mask: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
    """Applies the class head to the image features.

    Args:
      image_features: Feature tokens extracted by the image embedder.
      query_embeddings: Optional list of text (or image) embeddings. If no
        embeddings are provided, no logits will be computed and only the class
        embeddings for the image will be returned.
      query_mask: Must be provided with query_embeddings. A mask indicating
        which query embeddings are valid.

    Returns:
      A dictionary containing the class_embeddings and the pred_logits if
        query_embeddings and query_mask are provided.
    """
    return self._class_head(image_features, query_embeddings, query_mask)

  def mask_predictor(self,
                     image,
                     image_tokens,
                     boxes,
                     *,
                     true_boxes=None) -> Dict[str, jnp.ndarray]:
    """Predicts (cropped) segmentation masks from the image features.

    Args:
      image: Input image, for extracting low-level image features.
      image_tokens: High-level features from the image embedder.
      boxes: Predicted bounding boxes corresponding to the image tokens.
      true_boxes: For filtering mask head predictions during training.

    Returns:
      A dictionary containing the predicted segmentation masks. The mask at
        index i corresponds to the predicted box in `pred_boxes` at index i.
    """
    # TODO(b/215588365): Need local variable to work around pytype bug.
    mask_head_configs = self.mask_head_configs
    if mask_head_configs is None:
      raise ValueError('Must pass mask_head_configs to use mask head.')
    pred_masks = self._mask_head(
        image, image_tokens, boxes, true_boxes=true_boxes)
    batch_size = image_tokens.shape[0]
    mask_size = mask_head_configs.mask_size
    return {
        'pred_masks':
            jnp.reshape(pred_masks, (batch_size, -1, mask_size, mask_size))
    }

  def image_embedder(self, images: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Embeds images into feature maps.

    Args:
      images: images of shape (batch, input_size, input_size, 3), scaled to the
        input range defined in the config. Padding should be at the bottom right
        of the image.
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
    return text_features  # pytype: disable=bad-return-type  # jax-ndarray

  def __call__(self,
               inputs: jnp.ndarray,
               text_queries: jnp.ndarray,
               train: bool,
               *,
               true_boxes: Optional[jnp.ndarray] = None,
               debug: bool = False) -> Mapping[str, Any]:
    """Applies TextZeroShotDetectionModule on the input.

    Args:
      inputs: Images [batch_size, height, width, 3].
      text_queries: Queries to score boxes on. Queries starting with 0 stand for
        padding [batch_size=b, num_queries=q, max_query_length=l].
      train: Whether it is training.
      true_boxes: For filtering mask head predictions during training.
      debug: Unused.

    Returns:
      Outputs dict with items:
        pred_logits: Class logits [b, num_patches, num_queries].
        pred_boxes: Predicted bounding boxes [b, num_patches, 4].
        feature_map: Image embeddings 2d feature map [b, sp, sp, img_emb_dim].
    """
    del debug
    if not train and true_boxes is not None:
      raise ValueError('True boxes should only be supplied during training.')

    keep_tokens = None

    # Embed images:
    feature_map = self.image_embedder(inputs, train)
    b, h, w, d = feature_map.shape
    image_features = jnp.reshape(feature_map, (b, h * w, d))

    # Embed queries:
    query_embeddings = self.text_embedder(text_queries, train)
    # If first token is 0, then this is a padding query [b, q].
    query_mask = (text_queries[..., 0] > 0).astype(jnp.float32)

    outputs = {
        'feature_map': feature_map,
        'query_embeddings': query_embeddings,
    }

    # Get objectness scores:
    if self.objectness_head_configs is not None:
      outputs.update(self.objectness_predictor(image_features))

    # During training, sample top tokens by objectness:
    num_instances = image_features.shape[-2]
    top_k = self.body_configs.get('objectness_top_k', num_instances)
    if train and (0 < top_k < num_instances):
      if 'objectness_logits' not in outputs:
        raise ValueError('Need objectness head to sample by objectness.')
      outputs['objectness_logits'], keep_tokens = jax.lax.top_k(
          outputs['objectness_logits'], k=self.body_configs.objectness_top_k
      )
      image_features = jnp.take_along_axis(
          image_features, keep_tokens[..., None], axis=-2
      )

    # Classification [b, num_patches, num_queries]:
    outputs.update(
        self.class_predictor(image_features, query_embeddings, query_mask))

    # Predict boxes:
    outputs.update(
        self.box_predictor(
            image_features=image_features,
            feature_map=feature_map,
            keep_image_tokens=keep_tokens,
        )
    )

    # Predict masks:
    if self.mask_head_configs is not None:
      outputs.update(
          self.mask_predictor(
              inputs,
              image_features,
              outputs['pred_boxes'],
              true_boxes=true_boxes))

    return outputs

  def load(
      self, params: Params,
      init_config: ml_collections.ConfigDict) -> Params:
    """Loads backbone parameters for this model from a backbone checkpoint."""
    if init_config.get('codebase') == 'clip':
      # Initialize backbone parameters from an external codebase.
      params['backbone'] = self._embedder.load_backbone(
          params['backbone'], init_config.get('checkpoint_path'))
    else:
      # Initialize all parameters from a Scenic checkpoint.
      restored_train_state = checkpoints.restore_checkpoint(
          init_config.checkpoint_path, target=None)
      if 'optimizer' in restored_train_state:
        # Pre-Optax checkpoint:
        params = restored_train_state['optimizer']['target']
      else:
        params = restored_train_state['params']

      # Explicitly removing unused parameters after loading:
      params['class_head'].pop('padding', None)
      params['class_head'].pop('padding_bias', None)

    params = _fix_old_checkpoints(params)

    return params


class TextZeroShotDetectionModel(matching_base_models.ObjectDetectionModel):
  """OWL-ViT model for detection."""

  def build_flax_model(self) -> nn.Module:
    return TextZeroShotDetectionModule(
        body_configs=self.config.model.body,
        normalize=self.config.model.normalize,
        box_bias=self.config.model.box_bias)


class TextZeroShotDetectionModelWithMasks(
    matching_base_models.ObjectDetectionModelWithMasks):
  """ViT+ model for detection that also predicts masks."""

  def build_flax_model(self) -> nn.Module:
    return TextZeroShotDetectionModule(
        body_configs=self.config.model.body,
        mask_head_configs=self.config.model.mask_head,
        normalize=self.config.model.normalize,
        box_bias=self.config.model.box_bias)
