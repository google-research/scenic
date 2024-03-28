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

"""Optimization utils."""

from collections.abc import Callable
from typing import Any
from typing import Optional
from typing import Union

import optax


def _scale_by_learning_rate(
    learning_rate: optax.ScalarOrSchedule,
    flip_sign: bool = True,
) -> optax.GradientTransformation:
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def radamw(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:
  """The Rectified Adam optimiser.

  The adaptive learning rate in Adam has undesirably large variance in early
  stages of training, due to the limited number of training samples used to
  estimate the optimiser's statistics. Rectified Adam addresses this issue
  by analytically reducing the large variance.

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    threshold: the threshold for variance tractability.
    weight_decay: strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    mask: a tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      weight decay to, and `False` for those you want to skip. Note that the
      Adam gradient transformations are applied to all parameters.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return optax.chain(
      optax.scale_by_radam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, threshold=threshold),
      optax.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )

optax.radamw = radamw
