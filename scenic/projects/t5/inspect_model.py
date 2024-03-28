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

"""Logs T5 model parameter names and their shapes.

"""

from collections.abc import Sequence

from absl import app
from absl import flags

from scenic.common_lib import debug_utils
from scenic.projects.t5 import model
from t5x import checkpoints

_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path', None,
    'Path to a T5 checkpoint. This flag overrides "model_name".')
_MODEL_NAME = flags.DEFINE_enum(
    'model_name', 't5_1_1_small',
    ['t5_1_1_small', 't5_1_1_base', 't5_1_1_large', 't5_1_1_xl', 't5_1_1_xxl'],
    'The name of the T5 model to inspect.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  checkpoint_path = (
      _CHECKPOINT_PATH.value or model.CHECKPOINTS[_MODEL_NAME.value])
  params = checkpoints.load_t5x_checkpoint(checkpoint_path)['target']
  debug_utils.log_param_shapes(params)


if __name__ == '__main__':
  app.run(main)
