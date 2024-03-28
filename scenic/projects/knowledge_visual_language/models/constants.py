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

"""Constant value and types' definitions."""

from typing import Dict, Callable, Any, Tuple, Iterable

import jax.numpy as jnp

PyTree = Any
DType = jnp.dtype
JTensor = jnp.ndarray
JTensorDict = Dict[str, JTensor]
Batch = Dict[str, JTensor]
Shape = Iterable[int]
MetricFn = Callable[[JTensor, Batch], Dict[str, Tuple[float, int]]]
LossFn = Callable[[JTensorDict, Batch], Dict[str, float]]
Initializer = Callable[[JTensor, Shape, DType], JTensor]

