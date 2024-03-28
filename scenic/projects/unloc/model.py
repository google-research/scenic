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

"""Contains UnLoc models."""

from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.unloc import action_segmentation_base_model
from scenic.projects.unloc import encoders
from scenic.projects.unloc import heads
from scenic.projects.unloc import moment_retrieval_base_model
from scenic.projects.unloc import temporal_localization_base_model
from scenic.projects.unloc import video_text_fusion

# Mainly for unit tests.
_DEFAULT_UNLOC_CONFIG = ml_collections.ConfigDict({
    'model': ml_collections.ConfigDict({
        'video_tower_config': ml_collections.ConfigDict(
            {
                'modality_configs': {
                    'rgb': ml_collections.ConfigDict({
                        'encoder_name': 'clip_video_encoder',
                        'encoder_config': ml_collections.ConfigDict({
                            'num_classes': -1,
                            'image_encoder_config': ml_collections.ConfigDict({
                                'features': 8,
                                'num_layers': 2,
                                'num_heads': 2,
                                'classifier': 'token',
                                'patches': ml_collections.ConfigDict({
                                    'size': (4, 4, 1),
                                }),
                            }),
                            'temporal_encoding_config': ml_collections.ConfigDict({
                                'method': '3d_conv',
                                'kernel_init_method': (
                                    'central_frame_initializer'
                                ),
                            }),
                            'temporal_encoder_config': None,
                            'final_endpoint': 'temporal_tokens',
                            'classifier': 'gap',
                        }),
                    }),
                }
            }
        ),
        'text_tower_config': ml_collections.ConfigDict({
            'encoder_name': 'clip_text_encoder',
            'encoder_config': ml_collections.ConfigDict(
                dict(
                    vocab_size=100,
                    num_layers=2,
                    hidden_size=8,
                    num_heads=2,
                    classifier='eos',
                )
            ),
        }),
        'video_text_fusion_config': ml_collections.ConfigDict({
            'type': 'video_text_self_attention',
            'config': ml_collections.ConfigDict({
                'text_tower_classifier': 'token',
                'self_attention_encoder_config': ml_collections.ConfigDict({
                    'num_heads': 2,
                    'mlp_dim': 16,
                    'num_layers': 1,
                    'dropout_rate': 0.0,
                    'attention_dropout_rate': 0.0,
                    'stochastic_depth': 0.1,
                }),
                'use_all_text_tokens': False,
                'self_attention_encoder_name': 'transformer',
            }),
        }),
        'head_config': ml_collections.ConfigDict({
            'classification': ml_collections.ConfigDict({
                'type': 'linear_head',
                'config': ml_collections.ConfigDict(),
            }),
            'temporal_localization': ml_collections.ConfigDict({
                'type': 'query_dependent_localization_head',
                'config': ml_collections.ConfigDict({
                    'num_conv_layers': 3,
                    'kernel_size': 3,
                    'num_classes': -1,
                }),
            }),
            'highlight_detection': ml_collections.ConfigDict({
                'type': 'query_dependent_localization_head',
                'config': ml_collections.ConfigDict({
                    'num_conv_layers': 3,
                    'kernel_size': 3,
                    'num_classes': 1,
                }),
            }),
            'moment_retrieval': ml_collections.ConfigDict({
                'type': 'query_dependent_localization_head',
                'config': ml_collections.ConfigDict({
                    'num_conv_layers': 3,
                    'kernel_size': 3,
                }),
            }),
            'action_segmentation': ml_collections.ConfigDict({
                'type': 'linear_head',
                'config': ml_collections.ConfigDict(),
            }),
        }),
        'classifier': 'token',
        'num_classes': 10,
    }),
})


class VideoTextSingleTower(nn.Module):
  """Implements a video+text single-tower backbone.

  Attributes:
    num_classes: Number of output classes.
    video_tower_config: The config of the video tower.
    text_tower_config: The config of the text tower.
    video_text_fusion_config: The config of video+text fusion.
    classifier: 'gap' or 'token'
  """

  num_classes: int
  video_tower_config: ml_collections.ConfigDict
  text_tower_config: ml_collections.ConfigDict
  video_text_fusion_config: ml_collections.ConfigDict
  head_config: ml_collections.ConfigDict
  classifier: str = 'token'

  def setup(self):
    if self.video_tower_config.get('modality_configs') is None:
      # Ensure backward compatibility for single modality.
      self.video_encoder = encoders.ENCODERS[
          self.video_tower_config.encoder_name
      ](name='video_encoder', **self.video_tower_config.encoder_config)
      if self.video_tower_config.get('projection_size'):
        self.video_projection = nn.Dense(
            self.video_tower_config.projection_size,
            use_bias=self.video_tower_config.get('projection_use_bias', True),
            name='video_projection',
        )
    else:
      self.video_encoders = {  # pylint: disable=g-complex-comprehension
          modality_name: encoders.ENCODERS[modality_config.encoder_name](
              name=f'{modality_name}_encoder', **modality_config.encoder_config
          )
          for (
              modality_name,
              modality_config,
          ) in self.video_tower_config.modality_configs.items()
      }
      self.modality_ln = {  # pylint: disable=g-complex-comprehension
          modality_name: nn.LayerNorm(name=f'{modality_name}_ln')
          for (
              modality_name,
              _,
          ) in self.video_tower_config.modality_configs.items()
      }
      self.modality_projections = {  # pylint: disable=g-complex-comprehension
          modality_name: nn.Dense(
              modality_config.projection_size,
              use_bias=modality_config.get('projection_use_bias', True),
              name=f'{modality_name}_projection',
          )
          for (
              modality_name,
              modality_config,
          ) in self.video_tower_config.modality_configs.items()
          if modality_config.get('projection_size')
      }
      if self.video_tower_config.get('projection_size'):
        self.video_projection = nn.Dense(
            self.video_tower_config.projection_size,
            name='concat_video_projection',
        )
    if self.text_tower_config is not None:
      self.text_encoder = encoders.ENCODERS[
          self.text_tower_config.encoder_name
      ](name='text_encoder', **self.text_tower_config.encoder_config)
      if self.text_tower_config.get('projection_size'):
        self.text_projection = nn.Dense(
            self.text_tower_config.projection_size,
            use_bias=self.text_tower_config.get('projection_use_bias', True),
            name='text_projection',
        )
    self.fusion_model = video_text_fusion.FUSION_MODELS[
        self.video_text_fusion_config.type
    ](name='video_text_fusion', **self.video_text_fusion_config.config)
    self.head_models = {
        head_name: heads.HEADS[head_config.type](
            name=f'{head_name}_head', **head_config.config
        )
        for head_name, head_config in self.head_config.items()
    }

  def fuse_video_text(self,
                      video_tokens: jnp.ndarray,
                      text_tokens: jnp.ndarray,
                      task: str,
                      input_word_ids: Optional[jnp.ndarray] = None,
                      text_input_mask: Optional[jnp.ndarray] = None,
                      video_input_mask: Optional[jnp.ndarray] = None,
                      train: bool = True,
                      debug: bool = False) -> jnp.ndarray:
    """Fuses video and text tokens.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the image tokens.
      text_tokens: A 3D float tensor of shape (num_classes, sequence_length,
        channels) representing the text tokens.
      task: 'action_segmentation', classification', 'temporal_localization',
        'moment_retrieval' or 'highlight_detection'.
      input_word_ids: A 2D int tensor of shape (num_classes, sequence_length)
        representing the input word indices.
      text_input_mask: A 2D binary tensor of shape (batch_size, sequence_length)
        representing the mask of the text inputs.
      video_input_mask: A 2D binary tensor of shape (batch_size,
        sequence_length) representing the mask of the video inputs.
      train: Whether or not the model is under training.
      debug: Whether or not it is in debug mode.

    Returns:
      A 3D float tensor of shape (batch_size, num_classes, channels).

    Raises:
      ValueError if video_text_fusion_config.type is not supported.
    """
    video_tokens, text_tokens = self.fusion_model(video_tokens, text_tokens,
                                                  task, input_word_ids,
                                                  text_input_mask,
                                                  video_input_mask, train)
    return self.head_models[task](video_tokens, text_tokens, task, train)

  # TODO(xxman): support other multimodal fusion types.
  def encode_video(self,
                   inputs: Dict[str, Any],
                   train: bool = False,
                   debug: bool = False) -> jnp.ndarray:
    """Encodes video.

    We use a separate encoder to encode each modality and the output is obtained
    by concatenating encoded tokens in the channel dimension. We assume that
    all modality share the same sequence length.

    Args:
      inputs: Mappings from modality names to input data from each modality.
      train: Whether or not it is in training.
      debug: Whether or not it is in debug mode.

    Returns:
      A 3D float tensor of shape (batch, seq_length, channels) representing the
        concatenated encoded tokens.
    """

    if self.video_tower_config.get('modality_configs') is None:
      # Ensure backward compatibility.
      input_key = self.video_tower_config.get('input_key', 'rgb')
      video_tokens = self.video_encoder(
          inputs[input_key], train=train, debug=debug
      )
      if self.video_tower_config.get('freeze', False):
        video_tokens = jax.lax.stop_gradient(video_tokens)
      if self.video_tower_config.get('projection_size') is not None:
        video_tokens = self.video_projection(video_tokens)
      return video_tokens

    video_tokens = []
    for (
        modality_name,
        modality_config,
    ) in self.video_tower_config.modality_configs.items():
      tokens = self.video_encoders[modality_name](
          inputs[modality_name], train=train, debug=debug
      )
      if modality_config.get('freeze', False):
        tokens = jax.lax.stop_gradient(tokens)
      if modality_config.get('projection_size') is not None:
        tokens = self.modality_projections[modality_name](tokens)
      if modality_config.get('apply_post_encoder_layer_norm'):
        tokens = self.modality_ln[modality_name](tokens)
      video_tokens.append(tokens)
    video_tokens = jnp.concatenate(video_tokens, axis=-1)
    if self.video_tower_config.get('projection_size') is not None:
      video_tokens = self.video_projection(video_tokens)
    return video_tokens

  def encode_text(
      self,
      inputs: Dict[str, Any],
      task: str,
      train: bool = False,
      debug: bool = False,
  ) -> Optional[jnp.ndarray]:
    """Encodes text."""

    if self.text_tower_config is None:
      return None

    if task == 'moment_retrieval':
      # Merges all captions
      text_inputs = jax.tree_util.tree_map(
          lambda x: x.reshape((-1, x.shape[-1])), inputs['caption'])
    elif task == 'highlight_detection':
      input_key = self.text_tower_config.get('input_key', 'video_title')
      text_inputs = inputs[input_key]
    else:
      # The class names are the same in all batches.
      text_inputs = jax.tree_util.tree_map(lambda x: x[0],
                                           inputs['class_names'])
    text_tokens = self.text_encoder(text_inputs, train=train, debug=debug)
    if self.text_tower_config.get('freeze', False):
      text_tokens = jax.lax.stop_gradient(text_tokens)
    if self.text_tower_config.get('projection_size') is not None:
      text_tokens = self.text_projection(text_tokens)
    return text_tokens

  def __call__(self,
               inputs: Dict[str, Any],
               task: str = 'classification',
               dataset: str = '',
               train: bool = False,
               debug: bool = False):
    """Runs model inference.

    In this model, the video encoder, text encoder, and video-text fusion
    encoder are shared among all tasks. We build a different head for a
    different task and/or a different dataset. The head will have a unique name
    '{dataset}_{task}_head' if dataset is given. Otherwise, the head will have
    a name '{task}_head'.

    Args:
      inputs: Input dict containing the rgb frames or rgb embeddings and
        tokenized texts or text embeddings. RGB frames has a shape (batch_size,
        num_frames, height, width, channels) and RGB embeddings has a shape
        (batch_size, num_frames, channels). inputs['class_names'] or
        inputs['caption'] is dict of three elements whose keys are 'input_mask',
        'input_word_ids', and 'input_type_ids'.
      task: 'action_segmentation', classification', 'temporal_localization',
        'moment_retrieval' or 'highlight_detection'.
      dataset: The name of the dataset. The name will be used to create
        different heads for different datasets.
      train: Whether or not the model is under training.
      debug: Whether or not it is in debug mode.

    Returns:
      A 2D float tensor of shape (batch_size, num_classes) representing the
      logits for each class.
    """
    assert task in {
        'action_segmentation',
        'classification',
        'temporal_localization',
        'moment_retrieval',
        'highlight_detection',
    }

    input_word_ids = None
    text_input_mask = None
    if self.text_tower_config is not None:
      if task == 'moment_retrieval':
        # Merges all captions
        text_inputs = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, x.shape[-1])), inputs['caption']
        )
      elif task == 'highlight_detection':
        input_key = self.text_tower_config.get('input_key', 'video_title')
        text_inputs = inputs[input_key]
      else:
        # The class names are the same in all batches.
        text_inputs = jax.tree_util.tree_map(
            lambda x: x[0], inputs['class_names']
        )
      if (
          self.text_tower_config.get('input_type', 'tokenized_text')
          == 'tokenized_text'
      ):
        if task in {
            'action_segmentation',
            'temporal_localization',
            'classification',
        }:
          assert text_inputs['input_word_ids'].shape[0] == self.num_classes
        input_word_ids = text_inputs['input_word_ids']
        text_input_mask = text_inputs['input_mask']
    video_tokens = self.encode_video(inputs, train=train, debug=debug)
    text_tokens = self.encode_text(inputs, task=task, train=train, debug=debug)
    return self.fuse_video_text(
        video_tokens,
        text_tokens,
        task,
        input_word_ids,
        text_input_mask,
        video_input_mask=inputs.get('input_mask'),
        train=train,
        debug=debug)


class UnlocTemporalLocalizationModel(
    temporal_localization_base_model.TemporalLocalizationModel
):
  """Video-text single tower temporal localization model."""

  def build_flax_model(self) -> nn.Module:
    return VideoTextSingleTower(
        num_classes=self.dataset_meta_data['num_classes'],
        video_tower_config=self.config.model.video_tower_config,
        text_tower_config=self.config.model.text_tower_config,
        video_text_fusion_config=self.config.model.video_text_fusion_config,
        head_config=self.config.model.head_config,
        classifier=self.config.model.classifier)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _DEFAULT_UNLOC_CONFIG


class UnlocMomentRetrievalModel(
    moment_retrieval_base_model.MomentRetrievalModel
):
  """Video-text single tower moment retrieval model."""

  def build_flax_model(self) -> nn.Module:
    return VideoTextSingleTower(
        num_classes=self.dataset_meta_data['num_classes'],
        video_tower_config=self.config.model.video_tower_config,
        text_tower_config=self.config.model.text_tower_config,
        video_text_fusion_config=self.config.model.video_text_fusion_config,
        head_config=self.config.model.head_config,
        classifier=self.config.model.classifier)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _DEFAULT_UNLOC_CONFIG


class UnlocActionSegmentationModel(
    action_segmentation_base_model.ActionSegmentationModel
):
  """Video-text single tower action segmentation model."""

  def build_flax_model(self) -> nn.Module:
    return VideoTextSingleTower(
        num_classes=self.dataset_meta_data['num_classes'],
        video_tower_config=self.config.model.video_tower_config,
        text_tower_config=self.config.model.text_tower_config,
        video_text_fusion_config=self.config.model.video_text_fusion_config,
        head_config=self.config.model.head_config,
        classifier=self.config.model.classifier)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _DEFAULT_UNLOC_CONFIG


MODELS = {
    'unloc_action_segmentation': UnlocActionSegmentationModel,
    'unloc_highlight_detection': UnlocTemporalLocalizationModel,
    'unloc_moment_retrieval': UnlocMomentRetrievalModel,
    'unloc_temporal_localization': UnlocTemporalLocalizationModel,
}
