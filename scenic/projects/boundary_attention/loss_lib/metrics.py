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

"""Loss functions for the GT dataset."""


def gt_train_loss(model_output, batch, weights, loss_fn):
  """Computes the loss for the given model output and batch.

  Args:
    model_output: The output of the model.
    batch: The batch of data.
    weights: The weights of the batch.
    loss_fn: The loss function to use.

  Returns:
    The loss.
  """
  del weights

  return loss_fn.get_loss(model_output, batch)


def gt_standard_metric(model_output, batch, weights, loss_fn):
  """Computes the standard metric for the given model output and batch.

  Args:
    model_output: The output of the model.
    batch: The batch of data.
    weights: The weights of the batch.
    loss_fn: The loss function to use.

  Returns:
    The standard metric.
  """
  del weights

  return loss_fn.standard_metric(model_output, batch)


def gt_test_loss(model_output, batch, weights, loss_fn):
  """Computes the loss for the given model output and batch.

  Args:
    model_output: The output of the model.
    batch: The batch of data.
    weights: The weights of the batch.
    loss_fn: The loss function to use.

  Returns:
    The loss.
  """
  del weights

  return loss_fn.get_loss(model_output, batch)
