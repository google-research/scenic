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

"""Provides builders and loaders of T5X checkpoints.

Example usage:
```
from scenic.projects.t5 import model as t5
from scenic.projects.t5 import tokenizer as t5_tokenizer

model_name='t5_1_1_small'

scenic_model = t5.MODELS[model_name]()
scenic_loaded_state = t5.load_pretrained_weights(model_name)

scenic_model_bound = scenic_model.bind(scenic_loaded_state)

tokenizer = t5_tokenizer.build_dmvr_sp_model()

en_toks = np.array([tokenizer.string_to_indices(
    'Hi, <extra_id_0> is John', max_num_tokens=6, append_eos=True)])
de_toks = np.array([tokenizer.string_to_indices(
    '<extra_id_0> my name <extra_id_1>', max_num_tokens=7,
    prepend_bos=True, append_eos=True)])
de_toks[0, 0] = 0  # replace the BOS token to be 0 instaed of -1
inputs = (en_toks, de_toks[:, :-1], de_toks[:, 1:])

output = scenic_model_bound(*inputs)
tokenizer.indices_to_string([int(x) for x in np.argmax(output[0], -1)])
```
"""
from scenic.projects.t5 import layers
from t5x import checkpoints

# TODO(phseo): Implement beam search for general encoder-decoder models.

CHECKPOINTS = {
    't5_1_1_small':
        'gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000/',
    't5_1_1_base':
        'gs://t5-data/pretrained_models/t5x/t5_1_1_base/checkpoint_1000000/',
    't5_1_1_large':
        'gs://t5-data/pretrained_models/t5x/t5_1_1_large/checkpoint_1000000/',
    't5_1_1_xl':
        'gs://t5-data/pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000/',
    't5_1_1_xxl':
        'gs://t5-data/pretrained_models/t5x/t5_1_1_xxl/checkpoint_1000000/',
    'mt5_small':
        'gs://t5-data/pretrained_models/t5x/mt5_small/checkpoint_1000000/',
    'mt5_base':
        'gs://t5-data/pretrained_models/t5x/mt5_base/checkpoint_1000000/',
    'mt5_large':
        'gs://t5-data/pretrained_models/t5x/mt5_large/checkpoint_1000000/',
    'mt5_xl':
        'gs://t5-data/pretrained_models/t5x/mt5_xl/checkpoint_1000000/',
    'mt5_xxl':
        'gs://t5-data/pretrained_models/t5x/mt5_xxl/checkpoint_1000000/',
    'flan_t5_small':
        'gs://t5-data/pretrained_models/t5x/flan_t5_small/checkpoint_1198000',
    'flan_t5_base':
        'gs://t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000',
    'flan_t5_large':
        'gs://t5-data/pretrained_models/t5x/flan_t5_large/checkpoint_1164000',
    'flan_t5_xl':
        'gs://t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000',
    'flan_t5_xxl':
        'gs://t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000',
}


CONFIGS = {
    't5_1_1_small':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=512,
            num_heads=6,
            num_encoder_layers=8,
            num_decoder_layers=8,
            head_dim=64,
            mlp_dim=1024,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    't5_1_1_base':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=768,
            num_heads=12,
            num_encoder_layers=12,
            num_decoder_layers=12,
            head_dim=64,
            mlp_dim=2048,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    't5_1_1_large':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=1024,
            num_heads=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=2816,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    't5_1_1_xl':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=2048,
            num_heads=32,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=5120,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    't5_1_1_xxl':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=4096,
            num_heads=64,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=10240,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'mt5_small':
        dict(
            vocab_size=250112,
            dtype='bfloat16',
            emb_dim=512,
            num_heads=6,
            num_encoder_layers=8,
            num_decoder_layers=8,
            head_dim=64,
            mlp_dim=1024,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'mt5_base':
        dict(
            vocab_size=250112,
            dtype='bfloat16',
            emb_dim=768,
            num_heads=12,
            num_encoder_layers=12,
            num_decoder_layers=12,
            head_dim=64,
            mlp_dim=2048,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'mt5_large':
        dict(
            vocab_size=250112,
            dtype='bfloat16',
            emb_dim=1024,
            num_heads=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=2816,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'mt5_xl':
        dict(
            vocab_size=250112,
            dtype='bfloat16',
            emb_dim=2048,
            num_heads=32,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=5120,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'mt5_xxl':
        dict(
            vocab_size=250112,
            dtype='bfloat16',
            emb_dim=4096,
            num_heads=64,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=10240,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'flan_t5_small':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=512,
            num_heads=6,
            num_encoder_layers=8,
            num_decoder_layers=8,
            head_dim=64,
            mlp_dim=1024,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'flan_t5_base':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=768,
            num_heads=12,
            num_encoder_layers=12,
            num_decoder_layers=12,
            head_dim=64,
            mlp_dim=2048,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'flan_t5_large':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=1024,
            num_heads=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=2816,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'flan_t5_xl':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=2048,
            num_heads=32,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=5120,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
    'flan_t5_xxl':
        dict(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=4096,
            num_heads=64,
            num_encoder_layers=24,
            num_decoder_layers=24,
            head_dim=64,
            mlp_dim=10240,
            mlp_activations=('gelu', 'linear'),
            dropout_rate=0.0,
            logits_via_embedding=False),
}


def t5_1_1_small():
  return layers.T5(**CONFIGS['t5_1_1_small'])


def t5_1_1_base():
  return layers.T5(**CONFIGS['t5_1_1_base'])


def t5_1_1_large():
  return layers.T5(**CONFIGS['t5_1_1_large'])


def t5_1_1_xl():
  return layers.T5(**CONFIGS['t5_1_1_xl'])


def t5_1_1_xxl():
  return layers.T5(**CONFIGS['t5_1_1_xxl'])


def mt5_small():
  return layers.T5(**CONFIGS['mt5_small'])


def mt5_base():
  return layers.T5(**CONFIGS['mt5_base'])


def mt5_large():
  return layers.T5(**CONFIGS['mt5_large'])


def mt5_xl():
  return layers.T5(**CONFIGS['mt5_xl'])


def mt5_xxl():
  return layers.T5(**CONFIGS['mt5_xxl'])


def flan_t5_small():
  return layers.T5(**CONFIGS['flan_t5_small'])


def flan_t5_base():
  return layers.T5(**CONFIGS['flan_t5_base'])


def flan_t5_large():
  return layers.T5(**CONFIGS['flan_t5_large'])


def flan_t5_xl():
  return layers.T5(**CONFIGS['flan_t5_xl'])


def flan_t5_xxl():
  return layers.T5(**CONFIGS['flan_t5_xxl'])


MODELS = {
    't5_1_1_small': t5_1_1_small,
    't5_1_1_base': t5_1_1_base,
    't5_1_1_large': t5_1_1_large,
    't5_1_1_xl': t5_1_1_xl,
    't5_1_1_xxl': t5_1_1_xxl,
    'mt5_small': mt5_small,
    'mt5_base': mt5_base,
    'mt5_large': mt5_large,
    'mt5_xl': mt5_xl,
    'mt5_xxl': mt5_xxl,
    'flan_t5_small': flan_t5_small,
    'flan_t5_base': flan_t5_base,
    'flan_t5_large': flan_t5_large,
    'flan_t5_xl': flan_t5_xl,
    'flan_t5_xxl': flan_t5_xxl
}


def load_pretrained_weights(model_name, checkpoint_path=None):
  checkpoint_path = checkpoint_path or CHECKPOINTS.get(model_name)
  loaded_state = checkpoints.load_t5x_checkpoint(checkpoint_path)['target']
  loaded_state = {'params': {'t5_module': loaded_state}}
  return loaded_state
