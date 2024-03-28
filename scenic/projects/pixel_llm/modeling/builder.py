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

"""Builder functions for PixelLLM."""


import ml_collections
from scenic.projects.baselines.centernet.modeling import vitdet as centernet_vit
from scenic.projects.baselines.segment_anything.modeling import image_encoder as sam_vit
from scenic.projects.baselines.segment_anything.modeling import mask_decoder
from scenic.projects.baselines.segment_anything.modeling import prompt_encoder
from scenic.projects.pixel_llm.modeling import box_decoder
from scenic.projects.pixel_llm.modeling import eva02_vit
from scenic.projects.pixel_llm.modeling import layers
from scenic.projects.pixel_llm.modeling import mask_adapter
from scenic.projects.pixel_llm.modeling import point_predictor
from scenic.projects.pixel_llm.modeling import prompt_adapter
from scenic.projects.pixel_llm.modeling import t5_text_head
from scenic.projects.pixel_llm.modeling import text_decoder

ConfigDict = ml_collections.ConfigDict


def get_image_encoder(
    encoder_type: str,
    encoder_args: ConfigDict,
    param_name: str = 'image_encoder',
):
  """Returns an image encoder."""
  if encoder_type == 'eva02_vit':
    return eva02_vit.ViT(**encoder_args, name=param_name)
  elif encoder_type == 'sam_vit':
    return sam_vit.ImageEncoderViT(**encoder_args, name=param_name)
  elif encoder_type == 'centernet_vit':
    return centernet_vit.ViT(**encoder_args, name=param_name)
  elif encoder_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown encoder type {encoder_type}.')


def get_mask_decoder(
    decoder_type: str,
    decoder_args: ConfigDict,
    param_name: str = 'mask_decoder',
):
  """Returns a mask decoder."""
  if decoder_type == 'sam_mask_decoder':
    return mask_decoder.MaskDecoder(**decoder_args, name=param_name)
  elif decoder_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown decoder type {decoder_type}.')


def get_prompt_encoder(
    encoder_type: str,
    encoder_args: ConfigDict,
    param_name: str = 'prompt_encoder',
):
  """Returns an prompt encoder."""
  if encoder_type == 'sam_prompt_encoder':
    return prompt_encoder.PromptEncoder(**encoder_args, name=param_name)
  elif encoder_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown encoder type {encoder_type}.')


def get_prompt_adapter(
    adapter_type: str,
    adapter_args: ConfigDict,
    param_name: str = 'prompt_adapter',
):
  """Returns an prompt adapter."""
  if adapter_type == 'sam_prompt_adapter':
    return prompt_adapter.PromptAdaptor(**adapter_args, name=param_name)
  elif adapter_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown adapter type {adapter_type}.')


def get_point_predictor(
    predictor_type: str,
    predictor_args: ConfigDict,
    param_name: str = 'point_predictor',
):
  """Returns an point predictor."""
  if predictor_type == 'mlp_point_predictor':
    return point_predictor.MlpPointPredictor(
        **predictor_args, name=param_name
    )
  elif predictor_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown predictor type {predictor_type}.')


def get_text_decoder(
    decoder_type: str,
    vocab_size: int,
    decoder_args: ConfigDict,
    param_name: str = 'textual',
):
  """Returns a text decoder."""
  if decoder_type == 'git':
    return text_decoder.TransformerDecoderTextualHead(
        vocab_size=vocab_size,
        **decoder_args, name=param_name)
  elif 't5' in decoder_type:
    return t5_text_head.T5TextualHead(
        t5_model=decoder_type,
        **decoder_args,
        name=param_name)
  elif decoder_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown decoder type {decoder_type}.')


def get_box_decoder(
    decoder_type: str,
    decoder_args: ConfigDict,
    param_name: str = 'box_decoder',
):
  """Returns a box decoder."""
  if decoder_type == 'centernet2_det_decoder':
    return box_decoder.FpnCenterNet2(**decoder_args, name=param_name)
  elif decoder_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown decoder type {decoder_type}.')


def get_project_layers(
    project_layers_type: str,
    project_layers_args: ConfigDict,
    param_name: str = 'project_layers',
):
  """Returns a project layers."""
  if project_layers_type == 'linear':
    return layers.LinearProjectLayers(**project_layers_args, name=param_name)
  elif project_layers_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown project layers type {project_layers_type}.')


def get_mask_adapter(
    mask_adapter_type: str,
    mask_adapter_args: ConfigDict,
    param_name: str = 'mask_adapter',
):
  """Returns a mask adapter."""
  if mask_adapter_type == 'sam_mask_adapter':
    return mask_adapter.SamMaskAdaptor(**mask_adapter_args, name=param_name)
  elif mask_adapter_type == 'none':
    return None
  else:
    raise ValueError(f'Unknown mask adapter type {mask_adapter_type}.')
