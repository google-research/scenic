# Copyright 2021 The Scenic Authors.
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

"""Prints parameter content or shapes from a Scenic checkpoint.

If print_all_tensor_shapes=True, we print all tensor names and their shapes. For
example, if we provide a ViT checkpoint, the output will be as follows:

Transformer/posembed_input/pos_embedding: (1, 197, 768)
Transformer/encoder_norm/bias: (768,)
Transformer/encoder_norm/scale: (768,)
Transformer/encoderblock_0/LayerNorm_0/bias: (768,)
Transformer/encoderblock_0/LayerNorm_0/scale: (768,)
Transformer/encoderblock_0/LayerNorm_1/bias: (768,)
Transformer/encoderblock_0/LayerNorm_1/scale: (768,)
...
"""
import os
from typing import Any, Mapping, Sequence

from absl import app
from absl import flags
import flax
import numpy as np
from scenic.train_lib import pretrain_utils

_CHECKPOINT_PATH = flags.DEFINE_string('checkpoint_path', None,
                                       'Path to the scenic checkpoint.')
_PRINT_ALL_TENSOR_SHAPES = flags.DEFINE_bool(
    'print_all_tensor_shapes', True,
    'Whether or not to print all tensor shapes.')
_TENSOR_NAME = flags.DEFINE_string(
    'tensor_name', '',
    'Name of the tensor to print. The name is assumed to have nested scope, '
    'which is separated by `/`')


def log_shape(name: str, params: Mapping[str, Any]):
  if isinstance(params, np.ndarray):
    print(f'{name}: {params.shape}')
  else:
    for key, sub_params in params.items():
      log_shape(os.path.join(name, key), sub_params)


def log_param(scope_names: Sequence[str], params: Mapping[str, Any]):
  if len(scope_names) == 1:
    print(params)
  else:
    log_param(scope_names[1:], params.get(scope_names[0]))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  train_state = pretrain_utils.restore_pretrained_checkpoint(
      _CHECKPOINT_PATH.value, assert_exist=True)
  params = flax.core.unfreeze(train_state.optimizer['target'])
  if _PRINT_ALL_TENSOR_SHAPES.value:
    log_shape('', params)
  if _TENSOR_NAME.value:
    scope_names = _TENSOR_NAME.value.split('/')
    log_param(scope_names, params)


if __name__ == '__main__':
  app.run(main)
