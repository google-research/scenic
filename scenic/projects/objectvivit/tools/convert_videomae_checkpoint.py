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

r"""Convert VideoMAE checkpoint (by Tong et al) to a Scenic Train-state.

The model details are in https://arxiv.org/abs/2203.12602

To run this conversion script, we first need to save PyTorch model weights as
Numpy arrays.

```
import torch
import pickle
weights = torch.load('videomae_vitb_ssv2_2400e.pth', map_location='cpu')
weights_np = {k: v.numpy() for k, v in weights['model'].items()}
out = {'params': weights_np}
pickle.dump(out, open('videomae_vitb_ssv2_2400e.pkl', 'wb'))
```

Example command

python convert_videomae_checkpoint -- \
--model_version=B/16x2 \
--pytorch_data_path=videomae_vitb_ssv2_2400e.pkl \
--output_dir=videomae_vitb_ssv2_2400e/ \
"""

import pickle
from typing import Any, Dict, Union

from absl import app
from absl import flags
from absl import logging

import flax
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.objectvivit import model
from scenic.train_lib import train_utils

PyTree = Any
Array = Union[np.ndarray, jnp.ndarray]

flags.DEFINE_string(
    'model_version', 'B/16x2', 'ViT variant to load.')
flags.DEFINE_string(
    'pytorch_data_path',
    '',
    'Path containing PyTorch checkpoint data.')
flags.DEFINE_string(
    'output_dir',
    '',
    'Directory to write the Flax checkpoint to.')

FLAGS = flags.FLAGS

TRANSPOSE_PREFIXES = ('mlp.fc1.weight', 'mlp.fc2.weight',
                      'encoder_to_decoder.weight', 'decoder.head.weight',
                      'head.weight')


def get_vivit_config(variant: str) -> Dict[str, Any]:
  """Returns config for ViViT."""
  # Note that this config is used for testing that model outputs match. If using
  # the converted checkpoints, the same config settings should also be used.

  version, tubelet = variant.split('/')
  patch_s, patch_t = tubelet.split('x')

  config = {}
  config['temporal_encoding_config'] = ml_collections.ConfigDict()
  config['temporal_encoding_config'].method = '3d_conv'

  config['hidden_size'] = {'Ti': 192,
                           'S': 384,
                           'B': 768,
                           'L': 1024,
                           'H': 1280}[version]
  config['patches'] = ml_collections.ConfigDict()
  config['patches'].size = [int(patch_s), int(patch_s), int(patch_t)]
  config['num_heads'] = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config['mlp_dim'] = {'Ti': 768,
                       'S': 1536,
                       'B': 3072,
                       'L': 4096,
                       'H': 5120}[version]
  config['num_layers'] = {'Ti': 12,
                          'S': 12,
                          'B': 12,
                          'L': 24,
                          'H': 32}[version]
  config['attention_config'] = ml_collections.ConfigDict()
  config['attention_config'].type = 'spacetime'
  config['classifier'] = 'gap'
  config['attention_dropout_rate'] = 0.0
  config['dropout_rate'] = 0.0
  config['stochastic_droplayer_rate'] = 0.0
  config['positional_embedding'] = 'sinusoidal_1d'
  config['normalise_encoder_output'] = True
  config['use_approximate_gelu'] = False
  # Depends on the tubelet size, and if we are only reconstructing the central
  # frame.
  config['num_classes'] = 1
  return config


def maybe_transpose_weights(name: str, value: Array) -> Array:
  """Transposes PyTorch weight matrix if necessary for Jax."""

  for prefix in TRANSPOSE_PREFIXES:
    if prefix in name:
      return value.transpose()
  return value


def adapt_attention_weights(params: Dict[str, Array],
                            encoder_num_heads: int) -> Array:
  """Adapts PyTorch attention layers to Flax."""

  new_params = {}
  for name, parameter in params.items():

    if 'qkv.weight' in name or 'proj.weight' in name:
      if 'encoder' in name:
        num_heads = encoder_num_heads
      elif 'decoder' in name:
        continue
      else:
        raise ValueError(f'Unsupported parameter {name}.')

    if 'attn.qkv.weight' in name:
      # Un-merge the query-, key- and value-projections.
      query, key, value = np.split(parameter, 3, axis=0)
      query, key, value = query.transpose(), key.transpose(), value.transpose()
      d, _ = query.shape
      query = query.reshape(d, num_heads, -1)
      key = key.reshape(d, num_heads, -1)
      value = value.reshape(d, num_heads, -1)

      prefix = name.replace('qkv.weight', '')
      new_params[prefix + 'q_weight'] = query
      new_params[prefix + 'k_weight'] = key
      new_params[prefix + 'v_weight'] = value

      # We also need to reshape the bias parameters.
      q_bias = params[prefix + 'q_bias']
      q_bias = np.reshape(q_bias, (num_heads, -1))

      v_bias = params[prefix + 'v_bias']
      v_bias = np.reshape(v_bias, (num_heads, -1))

      if prefix + 'k_bias' in params:
        k_bias = params[prefix + 'v_bias']
        k_bias = np.reshape(k_bias, (num_heads, -1))
      else:
        logging.log_first_n(
            logging.INFO, 'No key bias. Initialising with zeros', 1)
        k_bias = np.zeros(q_bias.shape)

      new_params[prefix + 'q_bias'] = q_bias
      new_params[prefix + 'v_bias'] = v_bias
      new_params[prefix + 'k_bias'] = k_bias

    elif 'attn.proj.weight' in name:
      # Reshape the output projection with the number of heads.
      d, _ = parameter.shape
      output_proj = parameter.transpose().reshape((num_heads, -1, d))
      new_params[name] = output_proj

    else:
      new_params[name] = parameter

  return new_params  # pytype: disable=bad-return-type  # jax-ndarray


def rename_encoder_params(params: PyTree) -> PyTree:
  """Adds an encoder. prefix for finetuned models."""

  new_params = {}
  for name, value in params.items():
    if name.startswith('patch_embed') or name.startswith('blocks'):
      new_name = 'encoder.' + name
      new_params[new_name] = value
    else:
      new_params[name] = value

  return new_params


def convert_pytorch_parameters(
    params: Dict[str, Array],
    num_heads_encoder: int,
    num_encoder_layers: int) -> PyTree:
  """Adapt PyTorch model parameters to Jax ones.

  The steps are as follows.
    1. PyTorch models have query-, key- and value-projections fused, whereas in
       Flax, there are separate variables. Moreover, Flax has a separate axis
       for the number of heads. Whereas it is all fused in PyTorch.
    2. Transpose weights as necessary. As linear layer weights in PyTorch are
       transposed compared to nn.Dense layers in Flax.
    3. Rename model parameters according to our Scenic model.

  Args:
    params: A dictionary of PyTorch parameters.
    num_heads_encoder: The number of attention heads used in the transformer
      encoder.
    num_encoder_layers: The number of transformer layers in the encoder.

  Returns:
    A PyTree of Flax parameters, that can be used in a model.apply() call.
  """

  adapted_params = adapt_attention_weights(params, num_heads_encoder)

  for name, value in adapted_params.items():  # pytype: disable=attribute-error  # jax-ndarray
    adapted_params[name] = maybe_transpose_weights(name, value)

  # Rename parameters and move to final dictionary.
  unflattened_params = {}
  for name, value in adapted_params.items():  # pytype: disable=attribute-error  # jax-ndarray
    new_name = name
    # Rename specific parameters before renaming operations that affect multiple
    # parameters.

    # Input projection.
    # For convolution kernels, the PyTorch order is [c_out, c_in, t, h, w]. And
    # for Jax, it is [t, h, w, c_in, c_out].
    if new_name == 'encoder.patch_embed.proj.weight':
      new_name = 'embedding/kernel'
      value = value.transpose(2, 3, 4, 1, 0)

    if new_name == 'encoder.patch_embed.proj.bias':
      new_name = 'embedding/bias'

    # fc_norm only appears in finetuned models
    new_name = new_name.replace('fc_norm.weight',
                                'encoder_norm/scale')
    new_name = new_name.replace('fc_norm.bias',
                                'encoder_norm/bias')

    new_name = new_name.replace('encoder.norm.weight',
                                'Transformer/encoder_norm/scale')
    new_name = new_name.replace('encoder.norm.bias',
                                'Transformer/encoder_norm/bias')

    # The following appears only in pretrained models
    new_name = new_name.replace('decoder.head.weight',
                                'output_projection/kernel')
    new_name = new_name.replace('decoder.head.bias', 'output_projection/bias')
    # Whereas finetuned models have the following
    new_name = new_name.replace('head.weight', 'output_projection/kernel')
    new_name = new_name.replace('head.bias', 'output_projection/bias')

    # Rename "blocks.i to encoderblock_i"
    for i in range(num_encoder_layers):
      new_name = new_name.replace(f'blocks.{i}', f'encoderblock_{i}')

    # Rename transformer layer parameters.
    new_name = new_name.replace('encoder.', 'Transformer/')
    new_name = new_name.replace('decoder.', 'Decoder/')
    new_name = new_name.replace('attn', 'MultiHeadDotProductAttention_0')

    new_name = new_name.replace('q_weight', 'query/kernel')
    new_name = new_name.replace('q_bias', 'query/bias')

    new_name = new_name.replace('k_weight', 'key/kernel')
    new_name = new_name.replace('k_bias', 'key/bias')

    new_name = new_name.replace('v_weight', 'value/kernel')
    new_name = new_name.replace('v_bias', 'value/bias')

    new_name = new_name.replace('proj.weight', 'out/kernel')
    new_name = new_name.replace('proj.bias', 'out/bias')

    new_name = new_name.replace('norm1.weight', 'LayerNorm_0/scale')
    new_name = new_name.replace('norm1.bias', 'LayerNorm_0/bias')
    new_name = new_name.replace('norm2.weight', 'LayerNorm_1/scale')
    new_name = new_name.replace('norm2.bias', 'LayerNorm_1/bias')

    new_name = new_name.replace('mlp.fc1.weight', 'MlpBlock_0/Dense_0/kernel')
    new_name = new_name.replace('mlp.fc1.bias', 'MlpBlock_0/Dense_0/bias')
    new_name = new_name.replace('mlp.fc2.weight', 'MlpBlock_0/Dense_1/kernel')
    new_name = new_name.replace('mlp.fc2.bias', 'MlpBlock_0/Dense_1/bias')

    new_name = new_name.replace('.', '/')
    unflattened_params[new_name] = value

  params_tree = flax.traverse_util.unflatten_dict(unflattened_params, sep='/')
  return params_tree


def check_weights_different(initialisation: PyTree, loaded: PyTree,
                            ignore_key_bias: bool) -> bool:
  """Check all leaves are different in two PyTrees."""
  init_flat = flax.traverse_util.flatten_dict(initialisation, sep='/')
  loaded_flat = flax.traverse_util.flatten_dict(loaded, sep='/')

  for name in init_flat:
    if name not in loaded_flat:
      print(f'Variable {name} not in loaded model parameters.')
      continue
    if init_flat[name].shape != loaded_flat[name].shape:
      print(f'Variable {name} has incorrect shapes.')
      continue
    if np.allclose(init_flat[name], loaded_flat[name]):
      if ('MultiHeadDotProductAttention' in name and 'key/bias' in name and
          ignore_key_bias):
        # Include this special case, as the biases for the key projections are
        # often not used in PyTorch self-attention implementations.
        logging.info('Parameter %s did not change for initialisation', name)
        continue
      print(f'Variable {name} unchanged from initialisation.')

  return True


def load_pytorch_data(path: str):
  """Load checkpoint data from PyTorch."""

  with open(path, 'rb') as fp:
    data = pickle.load(fp)
  return data


def run(model_version: str, pytorch_data_path: str, output_dir: str):
  """Converts VideoMAE checkpoints to Jax, and checks for correctness."""

  # First initialise a ViT model and load weights.
  logging.info('Initialising ViT model')
  model_config = get_vivit_config(model_version)
  logging.info(model_config)
  vivit_model = model.ObjectViViT(**model_config)
  init_rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}

  rng = random.PRNGKey(0)
  batch, time, height, width, channels = 1, 16, 224, 224, 3
  input_shape = (batch, time, height, width, channels)
  inputs = random.normal(rng, shape=input_shape)

  _, params_jax = vivit_model.init_with_output(init_rngs, inputs, train=True)
  params_jax = flax.core.unfreeze(params_jax['params'])
  params_jax_init = jax.tree_util.tree_map(lambda x: x.copy(), params_jax)
  # Load VideoMAE model weights.
  logging.info('Loading VideoMAE ViViT weights')
  pretrained_pytorch_data = load_pytorch_data(pytorch_data_path)
  params_pytorch = pretrained_pytorch_data['params']

  # Transfer weights.
  logging.info('Transferring weights')
  params_jax = convert_pytorch_parameters(
      params_pytorch,
      model_config['num_heads'],
      model_config['num_layers'],
      )
  logging.info('Parameter summary:')
  params_jax = flax.core.freeze(params_jax)
  # Check weight transfer was correct.
  if check_weights_different(params_jax_init, params_jax, ignore_key_bias=True):
    logging.info('All model weights changed from initialisation.')

  # Save Scenic train state.
  logging.info('Saving converted checkpoint to %s', output_dir)
  train_state = train_utils.TrainState(
      global_step=0,
      params=params_jax,
      model_state={})
  train_utils.save_checkpoint(output_dir, train_state, overwrite=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run(FLAGS.model_version, FLAGS.pytorch_data_path, FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
