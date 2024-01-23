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

"""Image preprocessing library.

"""

from absl import logging
from scenic.dataset_lib.big_transfer import registry
from scenic.dataset_lib.big_transfer.preprocessing import ops as pp_ops
import tensorflow.compat.v1 as tf

TPU_SUPPORTED_DTYPES = [
    tf.bool, tf.int32, tf.int64, tf.bfloat16, tf.float32, tf.complex64,
    tf.uint32
]


def get_preprocess_fn(pp_pipeline, remove_tpu_dtypes=True, log_data=True):
  """Transform an input string into the preprocessing function.

  The minilanguage is as follows:

    fn1|fn2(arg, arg2,...)|...

  And describes the successive application of the various `fn`s to the input,
  where each function can optionally have one or more arguments, which are
  either positional or key/value, as dictated by the `fn`.

  The output preprocessing function expects a dictinary as input. This
  dictionary should have a key "image" that corresponds to a 3D tensor
  (height x width x channel).

  Args:
    pp_pipeline: A string describing the pre-processing pipeline. If empty or
      None, no preprocessing will be executed, but removing unsupported TPU
      dtypes will still be called if `remove_tpu_dtypes` is True.
    remove_tpu_dtypes: Whether to remove TPU incompatible types of data.
    log_data: Whether to log the data before and after preprocessing. Note:
      Remember set to `False` in eager mode to avoid too many log messages.

  Returns:
    preprocessing function.

  Raises:
    ValueError: if preprocessing function name is unknown
  """

  ops = []
  if pp_pipeline:
    for fn_name in pp_pipeline.split("|"):
      try:
        ops.append(registry.Registry.lookup(f"preprocess_ops.{fn_name}")())
      except SyntaxError as err:
        raise ValueError(f"Syntax error on: {fn_name}") from err

  def _preprocess_fn(data):
    """The preprocessing function that is returned."""

    # Validate input
    if not isinstance(data, dict):
      raise ValueError("Argument `data` must be a dictionary, "
                       "not %s" % str(type(data)))

    # Apply all the individual steps in sequence.
    if log_data:
      logging.info("Data before pre-processing:\n%s", data)
    for op in ops:
      data = op(data)

    if remove_tpu_dtypes:
      # Remove data that are TPU-incompatible (e.g. filename of type tf.string).
      for key in list(data.keys()):
        if data[key].dtype not in TPU_SUPPORTED_DTYPES:
          tf.logging.warning(
              "Removing key %s from data dict because its dtype %s is not in "
              " the supported dtypes: %s", key, data[key].dtype,
              TPU_SUPPORTED_DTYPES)
          data = pp_ops.get_delete_field(key)(data)
    if log_data:
      logging.info("Data after pre-processing:\n%s", data)
    return data

  return _preprocess_fn
