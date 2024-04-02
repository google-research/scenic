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

"""Common utilities for config files."""

import builtins
import ml_collections

BERT_TOKENIZER_PATH = '/path/to/bert-base-uncased/vocab.txt'

BERT_VOCAB_SIZE = 30522
SP_VOCAB_SIZE = 32128

ANET_TRAIN_SIZE = 8649
ANET_VAL_SIZE = 4267
ANET_PARA_VAL_SIZE = 2136
ANET_ANN_VID2SEQ_FORMAT_PATH = '/path/to/anet_val_vid2seq_format.json'
ANET_TRAIN_TFRECORD_PATH = '/path/to/anet_train.tfrecord@XX'
ANET_VAL_TFRECORD_PATH = '/path/to/anet_val.tfrecord@XX'
ANET_PARA_VAL_TFRECORD_PATH = '/path/to/anet_ae_val.tfrecord@XX'

YOUCOOK2_TRAIN_SIZE = 1333
YOUCOOK2_VAL_SIZE = 457
YOUCOOK2_ANN_VID2SEQ_FORMAT_PATH = '/path/to/youcook2_val_vid2seq_format.json'
YOUCOOK2_TRAIN_TFRECORD_PATH = '/path/to/youcook2_train.tfrecord@XX'
YOUCOOK2_VAL_TFRECORD_PATH = '/path/to/youcook2_val.tfrecord@XX'

VITT_TRAIN_SIZE = 4608
VITT_TEST_SIZE = 2301
VITT_ANN_VID2SEQ_FORMAT_PATH = '/path/to/vitt_val_vid2seq_format.json'
VITT_TRAIN_TFRECORD_PATH = '/path/to/vitt_train.tfrecord@XX'
VITT_TEST_TFRECORD_PATH = '/path/to/vitt_test.tfrecord@XX'


def control_flow(fn_flow, type=None, default=None):   # pylint: disable=redefined-builtin
  """Create a new field reference which is controlled by the fn_flow.

  Args:
    fn_flow (fct): Function of signature (default) -> resolved_value which
      compute the reference value
    type (type): Type of the field. At least one of type and default has
      to be defined. If not set, the type is deduced from the default value
    default (obj): The default value of the control flow. Is forwarded to the
      fn_flow.

  Returns:
    FieldReference: The field reference which lazy-execute the control flow.

  Raises:
    ValueError: If none of type and default are set.
  """
  type_ = type
  type = builtins.type

  if type_ is None and default is None:
    raise ValueError('At least of type or default has to be set.')

  if type_ is None:
    # If type is None, default won't be None
    type_ = type(default)

  if default is None:
    # Function op don't get applied if reference is None so initialize with
    # default constructor
    default = type_()

  if type_ is None:
    type_ = type(default)

  control_flow_op = ml_collections.config_dict._Op(fn=fn_flow, args=())   # pylint: disable=protected-access
  return ml_collections.FieldReference(
      # Function op don't get applied if reference is None so initialize with
      # default constructor
      default,
      field_type=type_,
      op=control_flow_op,
  )


def evaluate_lazily(fn):
  """Decorates fn with ref_util.control_flow to evaluate config fields lazily.

  Note that the decorated function must return a single output whose type does
  not depend on its inputs.

  Args:
    fn: Function of signature fn(*args, **kwargs) -> resolved_value, where args
      and kwargs are FieldReferences and resolved_value is a config value whose
      type and structure does not depend on the inputs to fn.

  Returns:
    A wrapped version of fn that returns a lazy FieldReference instead of an
    eagerly resolved value.
  """

  def lazy_fn(*args, **kwargs):
    all_args = list(args) + list(kwargs.values())
    if not all(isinstance(a, ml_collections.FieldReference) for a in all_args):
      raise ValueError(
          f'Please only pass FieldReferences to {fn.__name__}. For example, '
          f'instead of {fn.__name__}(config.key), use '
          f'{fn.__name__}(config.get_ref("key")).')

    def eager_fn(_):
      resolved_args = [a.get() for a in args]
      resolved_kwargs = {k: v.get() for k, v in kwargs.items()}
      return fn(*resolved_args, **resolved_kwargs)

    return control_flow(eager_fn, type=type(eager_fn(None)))

  return lazy_fn
