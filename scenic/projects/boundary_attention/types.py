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

"""Type aliases that are shared throughout the code."""

from typing import Callable, Dict, Iterable

import jax.numpy as jnp


# Aliases for custom types:
ArrayDict = Dict[str, jnp.ndarray]

MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, jnp.ndarray]]

LossFn = Callable[[ArrayDict, Dict[str, jnp.ndarray]], float]

Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]
