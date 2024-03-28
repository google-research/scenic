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

"""Contains head modules."""

from typing import List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.unloc import model_utils


class LinearHead(nn.Module):
  """Implements a linear head."""

  init_head_bias: float = 0.0
  output_dim: int = 1

  @nn.compact
  def __call__(
      self,
      video_tokens: jnp.ndarray,
      text_tokens: Optional[jnp.ndarray],
      task: str,
      train: bool,
  ) -> jnp.ndarray:
    """Builds a linear head.

    Args:
      video_tokens: A ND float tensor of shape (batch_size, ..., channels)
        representing the video tokens.
      text_tokens: A ND float tensor of shape (batch_size, ..., channels)
        representing the video tokens. Not used.
      task: 'action_segmentation'.
      train: Whether or not the model is under training. Not used.

    Returns:
      logits: A (N-1)D float tensor of shape (batch_size, ...) if output_dim ==
      1. Otherwise, A ND float tensor of shape (batch_size, ..., output_dim).
    """
    output = nn.Dense(
        self.output_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=jax.nn.initializers.constant(self.init_head_bias),
        name='output_projection',
    )(video_tokens)
    if self.output_dim == 1:
      return jnp.squeeze(output, axis=-1)
    return output


class ConvBlock(nn.Module):
  """Implements a multi-layer conv block."""

  output_dim: int
  num_conv_layers: int
  kernel_size: int = 3
  init_proj_bias: float = 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Builds a multi-layer convolution head.

    Args:
      x: Tokens in shape (batch_size, ..., num_frames, channels).

    Returns:
      Tensor in shape (batch_size, ..., num_frames, output_dim).
    """
    channels = x.shape[-1]
    for idx in range(self.num_conv_layers):
      x = nn.LayerNorm(name=f'ln{idx}')(x)
      x = nn.Conv(channels, [self.kernel_size], name=f'conv{idx}')(x)
      x = nn.relu(x)
    output = nn.Dense(
        self.output_dim,
        kernel_init=nn.initializers.xavier_normal(),
        bias_init=jax.nn.initializers.constant(self.init_proj_bias),
        name='output_projection',
    )(x)
    return output


class LocalizationHead(nn.Module):
  """Implements a localization head.

  We follow the head design in ActionFormer: https://arxiv.org/abs/2202.07925.
  In this head, we apply convolutions to generate per-frame logits for labels
  as well as the distance to the start/end times.

  Attributes:
    num_conv_layers: Number of convolution layers.
    kernel_size: Convolution kernel size.
    num_classes: Number of classes.
    init_classification_head_bias: Initial bias value for the classification
      head.
    init_regression_head_bias: Initial bias value for the regression head.
    distance_normalizer: We normalize the predicted displacements to be [0, 1].
      Options are: `relu_clip`, `sigmoid`, or 'relu'. When using `relu_clip`,
      the distances are first fed into relu() and then clipped to be [0, 1].
      When using `sigmoid`, the distances are normalized into [0, 1] using the
      `sigmoid` function. When using `relu`, distances are fed into a relu() but
      not normalized.
    weight_sharing: Whether or not to share the weights among decoders from
      different pyramid levels. If True, we will also learn a `scale` and a
      `bias` term for each level.
    feature_pyramid_config: Feature pyramid config.
    output_per_class_displacements: Whether or not to predict start/end time
      displacements for each class.
  """

  num_conv_layers: int
  kernel_size: int
  num_classes: int = 1
  init_classification_head_bias: float = 0.0
  init_regression_head_bias: float = 0.0
  distance_normalizer: str = 'relu'
  weight_sharing: bool = False
  feature_pyramid_config: Optional[ml_collections.ConfigDict] = None
  output_per_class_displacements: bool = True

  def _normalize_distance(self, distances: jnp.ndarray) -> jnp.ndarray:
    """Normalizes predicted distances to the start and end times."""
    if self.distance_normalizer == 'relu_clip':
      distances = nn.relu(distances)
      # We normalize the distances to be 0 and 1.
      return jnp.clip(distances, a_max=1.0)
    elif self.distance_normalizer == 'sigmoid':
      return nn.sigmoid(distances)
    elif self.distance_normalizer == 'relu':
      return nn.relu(distances)
    else:
      raise ValueError(
          f'Unknown distance_normalizer: {self.distance_normalizer}.'
      )

  def _build_with_weight_sharing(
      self,
      tokens: jnp.ndarray,
      classification_output_dim: int,
      regression_output_dim: int,
      pyramid_feature_axis: int,
  ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """Builds TAL head with shared decoders."""
    classification_head = ConvBlock(
        num_conv_layers=self.num_conv_layers,
        kernel_size=self.kernel_size,
        output_dim=classification_output_dim,
        init_proj_bias=self.init_classification_head_bias,
        name='ClassificationHead',
    )
    regression_head = ConvBlock(
        num_conv_layers=self.num_conv_layers,
        kernel_size=self.kernel_size,
        output_dim=regression_output_dim,
        init_proj_bias=self.init_regression_head_bias,
        name='RegressionHead',
    )
    if self.feature_pyramid_config is None:
      tokens_per_level = [tokens]
    else:
      # pytype: disable=attribute-error
      tokens_per_level = model_utils.split_pyramid_features(
          tokens,
          self.feature_pyramid_config.num_features_level0,
          len(self.feature_pyramid_config.feature_pyramid_levels),
          self.feature_pyramid_config.feature_pyramid_downsample_stride,
          axis=pyramid_feature_axis,
      )
      # pytype: enable=attribute-error
    all_classification_logits = []
    all_distances = []
    for level, x in enumerate(tokens_per_level):
      classification_logits = classification_head(x)
      if tokens.ndim == 4 and not self.output_per_class_displacements:
        x = jnp.mean(x, axis=1)
      distances = regression_head(x)
      scale = self.param(
          f'scale_{level}', nn.initializers.ones, (1,), distances.dtype
      )
      shift = self.param(
          f'shift_{level}', nn.initializers.zeros, (1,), distances.dtype
      )
      distances = self._normalize_distance(distances * scale + shift)
      all_classification_logits.append(classification_logits)
      all_distances.append(distances)
    return all_classification_logits, all_distances

  def _build_without_weight_sharing(
      self,
      tokens: jnp.ndarray,
      classification_output_dim: int,
      regression_output_dim: int,
      pyramid_feature_axis: int,
  ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """Builds TAL head with separate decoders."""
    if self.feature_pyramid_config is None:
      tokens_per_level = [tokens]
    else:
      # pytype: disable=attribute-error
      tokens_per_level = model_utils.split_pyramid_features(
          tokens,
          self.feature_pyramid_config.num_features_level0,
          len(self.feature_pyramid_config.feature_pyramid_levels),
          self.feature_pyramid_config.feature_pyramid_downsample_stride,
          axis=pyramid_feature_axis,
      )
      # pytype: enable=attribute-error
    all_classification_logits = []
    all_distances = []
    for level, x in enumerate(tokens_per_level):
      classification_logits = ConvBlock(
          num_conv_layers=self.num_conv_layers,
          kernel_size=self.kernel_size,
          output_dim=classification_output_dim,
          init_proj_bias=self.init_classification_head_bias,
          name=f'ClassificationHead_{level}',
      )(x)
      if tokens.ndim == 4 and not self.output_per_class_displacements:
        x = jnp.mean(x, axis=1)
      distances = ConvBlock(
          num_conv_layers=self.num_conv_layers,
          kernel_size=self.kernel_size,
          output_dim=regression_output_dim,
          init_proj_bias=self.init_regression_head_bias,
          name=f'RegressionHead_{level}',
      )(x)
      distances = self._normalize_distance(distances)
      all_classification_logits.append(classification_logits)
      all_distances.append(distances)
    return all_classification_logits, all_distances

  @nn.compact
  def __call__(
      self,
      video_tokens: jnp.ndarray,
      text_emb: jnp.ndarray,
      task: str,
      train: bool,
  ) -> jnp.ndarray:
    """Builds the temporal localization head.

    The head design is following ActionFormer, https://arxiv.org/abs/2202.07925.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, num_frames,
        channels) representing the fused video-text tokens.
      text_emb: Not used.
      task: Task name. Only supports 'temporal_localization' and
        'highlight_detection'.
      train: Whether or not it is in training.

    Returns:
      logits: A 3D float tensor in shape (batch_size, num_frames,
        num_classes * 3) if output_per_class_displacements = True, otherwise in
        shape (batch_size, num_frames, num_classes + 2) storing the logits of
        frame labels and the predicted distances to the start/end times.
    """
    del text_emb
    assert task == 'temporal_localization' or task == 'highlight_detection'
    assert video_tokens.ndim == 3
    bs, num_frames, _ = video_tokens.shape
    inputs = video_tokens
    if self.output_per_class_displacements:
      regression_output_dim = 2 * self.num_classes
    else:
      regression_output_dim = 2
    if self.weight_sharing:
      all_classification_logits, all_distances = (
          self._build_with_weight_sharing(
              inputs,
              classification_output_dim=self.num_classes,
              regression_output_dim=regression_output_dim,
              pyramid_feature_axis=1,
          )
      )
    else:
      all_classification_logits, all_distances = (
          self._build_without_weight_sharing(
              inputs,
              classification_output_dim=self.num_classes,
              regression_output_dim=regression_output_dim,
              pyramid_feature_axis=1,
          )
      )

    all_classification_logits = jnp.concatenate(
        all_classification_logits, axis=1
    )
    all_distances = jnp.concatenate(all_distances, axis=1)
    if self.output_per_class_displacements:
      all_classification_logits = all_classification_logits[..., jnp.newaxis]
      all_distances = all_distances.reshape(
          (bs, num_frames, self.num_classes, 2)
      )
    logits = jnp.concatenate(
        [all_classification_logits, all_distances], axis=-1
    )
    return logits.reshape((bs, num_frames, -1))


class QueryDependentLocalizationHead(LocalizationHead):
  """Implements a query dependent temporal localization head.

  The boundary regression takes into account both video and query information.
  """

  @nn.compact
  def __call__(
      self,
      video_tokens: jnp.ndarray,
      text_emb: jnp.ndarray,
      task: str,
      train: bool,
  ) -> jnp.ndarray:
    """Builds a query dependent localization head.

    Different from LocalizationHead, we assume video_tokens contain information
    from both the input video and text.

    Args:
      video_tokens: A 4D float tensor of shape (batch_size, num_texts,
        num_frames, channels) representing the fused video-text tokens.
      text_emb: Not used. A 3D float tensor of shape (batch_size, num_texts,
        channels) representing the text CLS token for each class. The second
        dimension is batch_size * max_num_captions if task=`moment_retrieval`.
        Otherwise, it is num_classes.
      task: Task name. 'temporal_localization', 'highlight_detection' or
        'moment_retrieval'.
      train: Whether or not it is in training.

    Returns:
      logits: In the case of 'temporal_localization' or 'highlight_detection',
        `logits` is a 3D float tensor in shape (batch_size, num_frames,
        num_classes * 3) if self.output_per_class_displacements = True,
        otherwise in shape (batch_size, num_frames, num_classes + 2). In the
        case of 'moment_retrieval', `logits` is a 4D tensor in shape
        (batch_size, num_texts, num_frames, 3) storing the logits of frame
        labels and the predicted distances to the start/end times.
    """
    assert video_tokens.ndim == 4 and text_emb.ndim == 3
    bs, _, num_frames, _ = video_tokens.shape

    if self.weight_sharing:
      all_classification_logits, all_distances = (
          self._build_with_weight_sharing(
              video_tokens,
              classification_output_dim=1,
              regression_output_dim=2,
              pyramid_feature_axis=2,
          )
      )
    else:
      all_classification_logits, all_distances = (
          self._build_without_weight_sharing(
              video_tokens,
              classification_output_dim=1,
              regression_output_dim=2,
              pyramid_feature_axis=2,
          )
      )

    all_classification_logits = jnp.concatenate(
        all_classification_logits, axis=2
    )
    if self.output_per_class_displacements:
      all_distances = jnp.concatenate(all_distances, axis=2)
    else:
      all_distances = jnp.concatenate(all_distances, axis=1)
    if task == 'moment_retrieval':
      return jnp.concatenate(
          [all_classification_logits, all_distances], axis=-1
      )
    elif task == 'temporal_localization' or task == 'highlight_detection':
      if self.output_per_class_displacements:
        logits = jnp.concatenate(
            [all_classification_logits, all_distances], axis=-1
        )
        # Transpose logits into (batch_size, num_frames, num_classes, 3)
        logits = jnp.transpose(logits, (0, 2, 1, 3))
        return logits.reshape((bs, num_frames, -1))
      else:
        # Changes logits shape into (batch_size, num_frames, num_classes).
        all_classification_logits = jnp.transpose(
            jnp.squeeze(all_classification_logits, axis=-1), [0, 2, 1]
        )
        return jnp.concatenate(
            [all_classification_logits, all_distances], axis=-1
        )
    else:
      raise ValueError(f'Unsupported task: {task}.')


HEADS = {
    'linear_head': LinearHead,
    'localization_head': LocalizationHead,
    'query_dependent_localization_head': QueryDependentLocalizationHead,
}
