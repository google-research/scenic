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

"""Helper functions for exporting JAX models to Tensorflow SavedModels."""

from typing import Any, Callable, Sequence, Optional, Union

from jax.experimental import jax2tf
import tensorflow as tf
import tree as dm_tree

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
# A PyTree is a nested dictionary where the leaves are `jnp.ndarray`.
# TODO(aarnab): Fix type annotation once ready.
PyTree = Any


def convert_and_save_model(
    jax_fn: Callable[[PyTree, PyTree], PyTree],
    params: PyTree,
    model_dir: str,
    *,
    input_signatures: Union[
        Sequence[tf.TensorSpec],
        Sequence[Sequence[tf.TensorSpec]],
        Sequence[dict[str, tf.TensorSpec]],
    ],
    polymorphic_shapes: Optional[
        Union[str, jax2tf.PolyShape, dict[str, str]]
    ] = None,
    with_gradient: bool = False,
    enable_xla: bool = True,
    compile_model: bool = True,
    saved_model_options: Optional[tf.saved_model.SaveOptions] = None,
    native_serialization: Optional[str | bool] = "default",
    native_serialization_platforms: Sequence[str] | None = ("cpu", "tpu")):
  """Converts a JAX function and saves a SavedModel.

  We assume that the JAX model consists of a prediction function and trained
  parameters, and the computation graph of the function is saved separately from
  the parameters. Saving the graph separately from the parameters reduces
  the size of the Tensorflow `GraphDef`, and enables finetuning of model
  parameters too.

  To use this function, a JAX model must be converted to a function of two
  arguments, the model parameters and the input.
  For a Scenic model, this corresponds to:
  ```
  params = train_state.optimizer.target
  flax_model = model.flax_model
  def _predict_fn(params, input_data):
    return flax_model.apply({'params': params}, input_data, train=False)
  ```

  Args:
    jax_fn: A JAX function taking two arguments, the parameters and the inputs.
      Both arguments may be (nested) tuples/lists/dictionaries of `np.ndarray`.
      It is necessary to be able to JIT-compile this function (ie run
      `jax.jit` on it).
    params: The parameters, to be used as first argument for `jax_fn`. These
      must be (nested) tuples/lists/dictionaries of `np.ndarray`, and will be
      saved as the variables of the SavedModel.
    model_dir: The directory where the model should be saved.
    input_signatures: The input signatures for the second argument of `jax_fn`
      (the input). A signature must be a `tensorflow.TensorSpec` instance, or a
      (nested) tuple/list/dictionary thereof with a structure matching the
      second argument of `jax_fn`. The first input_signature will be saved as
      the default serving signature. The additional signatures will be used
      only to ensure that the `jax_fn` is traced and converted to TF for the
      corresponding input shapes.
    polymorphic_shapes: If given then it will be used as the
      `polymorphic_shapes` argument to `jax2tf.convert` for the second parameter
      of `jax_fn`. In this case, a single `input_signatures` is supported, and
      should have `None` in the polymorphic dimensions. This is required, for
      example, to have models with dynamic batch sizes.
    with_gradient: Whether the SavedModel should support gradients. If `True`,
      then a custom gradient is saved. If `False`, then a
      `tf.raw_ops.PreventGradient` is saved to error if a gradient is attempted.
      (At the moment due to a bug in SavedModel, custom gradients are not
      supported.)
    enable_xla: Whether the jax2tf converter is allowed to use TF XLA ops. If
      `False`, the conversion tries harder to use purely TF ops and raises an
      exception if it is not possible.
    compile_model: Use TensorFlow jit_compiler on the SavedModel. This
      is needed if the SavedModel will be used for TensorFlow serving.
    saved_model_options: Options to pass to `savedmodel.save`.
    native_serialization: Serialize the JAX function natively to
      StableHLO with compatibility guarantees. This makes it easier to have
      confidence that the code executed when calling this function from
      TensorFlow is exactly the same as JAX would run natively. See
      jax2tf.convert() for details.
    native_serialization_platforms: When the "native_serialization" flag is
      used, the platforms that it will be serialised to. Must be a tuple of
      strings, including a subset of: ['cpu', 'cuda', 'rocm', 'tpu'].
      'None', specifies the JAX default backend on the machine where the
      lowering is done.

  Raises:
    ValueError: If at least one input signature is not defined. However, if
    `polymorphic_shapes` is given, then only one input signature is supported.
  """
  if not input_signatures:
    raise ValueError("At least one input_signature must be given.")
  if polymorphic_shapes is not None and len(input_signatures) > 1:
    raise ValueError("For shape-polymorphic conversion a single "
                     "input_signature is supported.")
  tf_fn = jax2tf.convert(
      jax_fn,
      with_gradient=with_gradient,
      polymorphic_shapes=[None, polymorphic_shapes],
      enable_xla=enable_xla,
      native_serialization=native_serialization,
      native_serialization_platforms=native_serialization_platforms)

  def get_tf_variable(path, param):
    return tf.Variable(param, trainable=with_gradient, name="/".join(path))

  param_vars = dm_tree.map_structure_with_path(
      # Due to a bug in SavedModel it is not possible to use `tf.GradientTape`
      # on a function converted with jax2tf and loaded from SavedModel. Thus, we
      # mark the variables as non-trainable to ensure that users of the
      # SavedModel will not try to fine tune them.
      get_tf_variable, params)
  tf_graph = tf.function(
      lambda inputs: tf_fn(param_vars, inputs),
      autograph=False,
      jit_compile=compile_model)

  # This signature is needed for TensorFlow Serving use.
  signatures = {
      tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          tf_graph.get_concrete_function(input_signatures[0])
  }

  for input_signature in input_signatures[1:]:
    # If there are more signatures, trace and cache a TF function for each one.
    tf_graph.get_concrete_function(input_signature)
  wrapper = _ReusableSavedModelWrapper(tf_graph, param_vars)

  if saved_model_options:
    saved_model_options.function_aliases = {"inference_func": tf_graph}
  else:
    saved_model_options = tf.saved_model.SaveOptions(
        function_aliases={"inference_func": tf_graph}
    )

  if with_gradient:
    saved_model_options.experimental_custom_gradients = True

  tf.saved_model.save(
      wrapper, model_dir, signatures=signatures, options=saved_model_options
  )


class _ReusableSavedModelWrapper(tf.train.Checkpoint):
  """Wraps a function and its parameters for saving to a SavedModel.

  Implements the interface described at
  https://www.tensorflow.org/hub/reusable_saved_models.
  """

  def __init__(self, tf_graph: Callable[[PyTree], PyTree], param_vars: PyTree):
    """Constructor.

    Args:
      tf_graph: A `tf.function` taking one argument (the inputs), which can be
        be tuples/lists/dictionaries of `np.ndarray` or tensors. The function
        may have references to the `tf.Variables` in `param_vars`.
      param_vars: The parameters, as tuples/lists/dictionaries of
        `tf.Variable`, to be saved as the variables of the SavedModel.
    """
    super().__init__()
    self.variables = tf.nest.flatten(param_vars)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    self.__call__ = tf_graph
