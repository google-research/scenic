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

"""Contains config utils."""

import ml_collections

CLIP_IMAGE_ENCODER_CONFIGS = {
    'B/16':
        dict(
            features=768,
            num_layers=12,
            num_heads=12,
            stochastic_depth=0.0,
            classifier='token',
        ),
    'B/32':
        dict(
            features=768,
            num_layers=12,
            num_heads=12,
            stochastic_depth=0.0,
            classifier='token',
        ),
    'L/14':
        dict(
            features=1024,
            num_layers=24,
            num_heads=16,
            stochastic_depth=0.0,
            classifier='token',
        ),
}

CLIP_TEXT_ENCODER_CONFIGS = {
    'B/16':
        dict(
            vocab_size=49408,
            num_layers=12,
            hidden_size=512,
            num_heads=8,
            classifier='eos',
        ),
    'B/32':
        dict(
            vocab_size=49408,
            num_layers=12,
            hidden_size=512,
            num_heads=8,
            classifier='eos',
        ),
    'L/14':
        dict(
            vocab_size=49408,
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            classifier='eos',
        ),
}

T5_ENCODER_CONFIGS = {
    't5_1_1_small': dict(
        vocab_size=32128,
        emb_dim=512,
        num_heads=6,
        num_encoder_layers=8,
        num_decoder_layers=8,
        head_dim=64,
        mlp_dim=1024,
        dropout_rate=0.0,
        classifier='gap',
    ),
    't5_1_1_base': dict(
        vocab_size=32128,
        emb_dim=768,
        num_heads=12,
        num_encoder_layers=12,
        num_decoder_layers=12,
        head_dim=64,
        mlp_dim=2048,
        dropout_rate=0.0,
        classifier='gap',
    ),
    't5_1_1_large': dict(
        vocab_size=32128,
        emb_dim=1024,
        num_heads=16,
        num_encoder_layers=24,
        num_decoder_layers=24,
        head_dim=64,
        mlp_dim=2816,
        dropout_rate=0.0,
        classifier='gap',
    ),
    't5_1_1_xl': dict(
        vocab_size=32128,
        emb_dim=2048,
        num_heads=32,
        num_encoder_layers=24,
        num_decoder_layers=24,
        head_dim=64,
        mlp_dim=5120,
        dropout_rate=0.0,
        classifier='gap',
    ),
    't5_1_1_xxl': dict(
        vocab_size=32128,
        emb_dim=4096,
        num_heads=64,
        num_encoder_layers=24,
        num_decoder_layers=24,
        head_dim=64,
        mlp_dim=10240,
        dropout_rate=0.0,
        classifier='gap',
    ),
}


def parse_t5_encoder_config(variant: str) -> ml_collections.ConfigDict:
  """Parses T5 parameters."""
  return ml_collections.ConfigDict(T5_ENCODER_CONFIGS[variant])


def parse_image_encoder_config(variant: str) -> ml_collections.ConfigDict:
  """Parse model configs from an encoded text.

  The model is encoded in the format of 'vit_version/patch_size'. For example,
  'B/16x2' is the Base model trained on tubelets of size 16x16x2.

  Args:
    variant: a str encoding the model structure.

  Returns:
     model configs.
  """
  version, tublet_size = variant.split('/')
  patch_size, num_frames = tublet_size.split('x')
  version = '/'.join([version, patch_size])
  num_frames = int(num_frames)
  patch_size = int(patch_size)
  config = CLIP_IMAGE_ENCODER_CONFIGS[version]
  config['patches'] = {'size': (patch_size, patch_size, num_frames)}
  return ml_collections.ConfigDict(config)


def parse_text_encoder_config(variant: str) -> ml_collections.ConfigDict:
  """Parse model configs from an encoded text.

  The model is encoded in the format of 'vit_version/patch_size'. For example,
  'B/16x2' is the Base model trained on tubelets of size 16x16x2. The temporal
  dimension of the tubelet is ignored for text encoders.

  Args:
    variant: a str encoding the model structure.

  Returns:
     model configs.
  """
  version, tublet_size = variant.split('/')
  patch_size, _ = tublet_size.split('x')
  version = '/'.join([version, patch_size])
  config = CLIP_TEXT_ENCODER_CONFIGS[version]
  return ml_collections.ConfigDict(config)
