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

