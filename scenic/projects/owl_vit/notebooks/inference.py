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

"""Code for running (interactive) inference with OWL-ViT models."""

import dataclasses
import functools
from typing import Any, Dict, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import box_utils
from scenic.projects.owl_vit.notebooks import numpy_cache
from scipy import special as sp_special
from skimage import transform as skimage_transform
import tensorflow as tf

sigmoid = sp_special.expit  # Sigmoid is a more familiar name.
QUERY_PAD_BIN_SIZE = 50


@dataclasses.dataclass
class Model:
  """Wraps an OWL-ViT FLAX model for convenient inference.

  All public methods apply to a single example and take and return Numpy arrays.

  Attributes:
    config: ConfigDict with model configuration.
    module: OWL-ViT Flax module.
    variables: Variable dict to be used with module.apply.
  """

  config: ml_collections.ConfigDict
  module: nn.Module
  variables: Dict[str, Any]

  def __eq__(self, other):
    if isinstance(other, Model):
      return (self.config.init_from.checkpoint_path ==
              other.config.init_from.checkpoint_path)

  def __hash__(self):
    return hash(self.config.init_from.checkpoint_path)

  def warm_up(self):
    """Runs the model on a dummy example to trigger compilation."""
    image = np.zeros((128, 64, 3), dtype=np.uint8)
    queries = ('dummy',)
    query_embeddings = self.embed_text_queries(queries)
    self.get_scores(image, query_embeddings, len(queries))

  @numpy_cache.lru_cache(maxsize=100)
  def preprocess_image(self, image: np.ndarray) -> np.ndarray:
    """Preprocesses a uint8 image to the format required by the model."""
    if image.dtype != np.uint8:
      raise ValueError(f'Image should be uint8, got {image.dtype}')
    image = image.astype(np.float32) / 255.0

    # Pad to square with gray pixels on bottom and right:
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    # Resize to model input size:
    return skimage_transform.resize(
        image_padded, (self.config.dataset_configs.input_size,
                       self.config.dataset_configs.input_size),
        anti_aliasing=True)

  @numpy_cache.lru_cache(maxsize=100)
  def embed_image(
      self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeds image and returns image embeddings and boxes.

    Args:
      image: Single uint8 Numpy image of any size. Will be converted to float
        and resized before passing it to the model.

    Returns:
      Numpy arrays containing image features, class embeddings, and predicted
      boxes.
    """
    image = self.preprocess_image(image)
    out = self._embed_image_jitted(image[None, ...])
    return jax.tree_util.tree_map(lambda x: np.array(x[0]), out)

  @numpy_cache.lru_cache(maxsize=1000)
  def embed_text_queries(self, queries: Tuple[str, ...]) -> np.ndarray:
    """Embeds text queries.

    Args:
      queries: Tuple of query strings.

    Returns:
      Numpy arrays containing query embeddings.
    """
    tokenized = np.array([
        self.module.apply(self.variables, q, method=self.module.tokenize)
        for q in queries])
    # Pad queries to avoid re-compilation:
    n = len(queries)
    num_pad = int(np.ceil(n / QUERY_PAD_BIN_SIZE) * QUERY_PAD_BIN_SIZE) - n
    tokenized = tf.pad(tokenized, [[0, num_pad], [0, 0]]).numpy()
    return np.array(self._embed_texts_jitted(tokenized[None, ...])[0])

  @numpy_cache.lru_cache(maxsize=100)
  def embed_image_query(
      self,
      query_image: np.ndarray,
      query_box_yxyx: Tuple[float, float, float, float],
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts image features in the region of `desired_boxes_yxyx`.


    This works by taking the features of a bounding box with high IOU
    to the desired_boxes_yxyx and returning the corresponding embedding
    features.

    Args:
      query_image: Image showing the example object to be used as query.
      query_box_yxyx: A single bounding box around the example object, in the
        TensorFlow format (y_min, x_min, y_max, x_max), normalized to [0, 1],
        for which to extract a query embedding.

    Returns:
      Queryable features and index of the predicted box whose features were
      selected.
    """
    _, class_embeddings, pred_boxes = self.embed_image(query_image)
    ious = box_utils.box_iou(
        np.array(query_box_yxyx)[None, ...],
        box_utils.box_cxcywh_to_yxyx(pred_boxes, np),
        np_backbone=np)[0][0]

    # If there are no overlapping boxes, fall back to generalized IoU:
    if np.all(ious == 0.0):
      ious = box_utils.generalized_box_iou(
          np.array(query_box_yxyx)[None, ...],
          box_utils.box_cxcywh_to_yxyx(pred_boxes, np),
          np_backbone=np)[0]

    # Use an adaptive threshold such that all boxes within 80% of the best IoU
    # are included:
    iou_thresh = np.max(ious) * 0.8

    # Select class_embeddings that are above the IoU threshold:
    selected_inds = (ious >= iou_thresh).nonzero()[0]
    assert selected_inds.size
    selected_embeddings = class_embeddings[selected_inds]

    # Due to the DETR style bipartite matching loss, only one embedding
    # feature for each object is "good" and the rest are "background." To find
    # the one "good" feature we use the heuristic that it should be dissimilar
    # to the mean embedding.
    mean_embedding = np.mean(class_embeddings, axis=0)
    mean_sim = np.einsum('d,id->i', mean_embedding, selected_embeddings)

    # Find box with lowest overall similarity:
    best_box_ind = selected_inds[np.argmin(mean_sim)]

    return class_embeddings[best_box_ind], best_box_ind  # pytype: disable=bad-return-type  # jax-ndarray

  def get_scores(
      self,
      image: np.ndarray,
      query_embeddings: np.ndarray,
      num_queries: int,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Scores image features against queries.

    Args:
      image: Single uint8 Numpy image of any size. Will be converted to float
        and resized before passing it to the model.
      query_embeddings: Text- or image-derived queries.
      num_queries: Number of true queries, in case embeddings are padded.

    Returns:
      Index and score of the top query for each predicted box in the image.
    """
    image_features, _, _ = self.embed_image(image)
    out = self._predict_classes_jitted(
        image_features=image_features[None, ...],
        query_embeddings=query_embeddings[None, ...])
    logits = np.array(out['pred_logits'])[0, :, :num_queries]  # Remove padding.
    top_query_ind = np.argmax(logits, axis=-1)
    scores = sigmoid(np.max(logits, axis=-1))
    return top_query_ind, scores

  @functools.partial(jax.jit, static_argnums=(0,))
  def _embed_image_jitted(
      self, image: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Embeds image and returns image features, class embeddings, and boxes."""
    feature_map = self.module.apply(
        self.variables,
        images=image,
        train=False,
        method=self.module.image_embedder)
    b, _, _, c = feature_map.shape
    features = jnp.reshape(feature_map, (b, -1, c))
    pred_boxes = self.module.apply(
        self.variables, image_features=features, feature_map=feature_map,
        method=self.module.box_predictor)['pred_boxes']
    class_embeddings = self.module.apply(
        self.variables,
        image_features=features,
        query_embeddings=None,
        method=self.module.class_predictor)['class_embeddings']
    return features, class_embeddings, pred_boxes

  @functools.partial(jax.jit, static_argnums=(0,))
  def _predict_classes_jitted(
      self,
      image_features: jnp.ndarray,
      query_embeddings: jnp.ndarray,
  ) -> Dict[str, jnp.ndarray]:
    return self.module.apply(
        self.variables,
        image_features=image_features,
        query_embeddings=query_embeddings,
        method=self.module.class_predictor)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _embed_texts_jitted(self, queries: jnp.ndarray) -> jnp.ndarray:
    return self.module.apply(self.variables,
                             text_queries=queries,
                             train=False,
                             method=self.module.text_embedder)
