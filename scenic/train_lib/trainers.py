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

"""Registry for the available trainers."""

from scenic.train_lib import classification_trainer
from scenic.train_lib.transfer import transfer_trainer

ALL_TRAINERS = {
    'classification_trainer': classification_trainer.train,
    'transfer_trainer': transfer_trainer.train,
}


def get_trainer(train_fn_name):
  """Get the corresponding trainer function.

  The returned train function has the following API:
  ```
    train_state, train_summary, eval_summary = train_fn(
      rng, model_cls, dataset, config, workdir, summary_writer)
  ```
  Where the train_state is a checkpointable state of training and train_summary,
  and eval_summary are python dictionary that contains metrics.

  Args:
    train_fn_name: str; Name of the train_fn_name, e.g.
      'classification_trainer'.

  Returns:
    The train function.
  Raises:
    ValueError if train_fn_name is unrecognized.
  """
  if train_fn_name not in ALL_TRAINERS.keys():
    raise ValueError('Unrecognized trainer: {}'.format(train_fn_name))
  return ALL_TRAINERS[train_fn_name]
