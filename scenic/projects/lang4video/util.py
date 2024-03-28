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

"""General utilities."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
import copy
import types
from typing import Any
from typing import Optional
from typing import TypeVar
from typing import Union

from dmvr import tokenizers
import jax
import ml_collections

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


def batch_sequence(seq: Sequence[T], n: int) -> Iterator[Sequence[T]]:
  for i in range(0, len(seq), n):
    yield seq[i:i + n]


def _compute_field_reference_replace_map(
    config: Union[ml_collections.ConfigDict, Mapping[str, Any], Iterable[Any]],
    new_config: Union[ml_collections.ConfigDict, Mapping[str, Any],
                      Iterable[Any]],
    field_reference_replace_map: MutableMapping[int,
                                                ml_collections.FieldReference],
) -> None:
  """Computes the mapping from the old field references to the new ones."""
  if isinstance(config, (ml_collections.ConfigDict, Mapping)):
    fields = config._fields if isinstance(  # pylint: disable=protected-access
        config, ml_collections.ConfigDict) else config
    new_fields = new_config._fields if isinstance(  # pylint: disable=protected-access
        new_config, ml_collections.ConfigDict) else new_config
    it = ((v, new_fields[k]) for k, v in fields.items())
  else:
    it = zip(config, new_config)

  for v, new_v in it:
    if isinstance(v, ml_collections.FieldReference):
      field_reference_replace_map[id(v)] = new_v
    elif isinstance(v, (ml_collections.ConfigDict,
                        Iterable)) and not isinstance(config, (str, bytes)):
      _compute_field_reference_replace_map(v, new_v,
                                           field_reference_replace_map)


def create_cell_object(x: Any):
  return (lambda y: lambda: y)(x).__closure__[0]


def replace_in_config(
    obj: Union[ml_collections.ConfigDict, Mapping[str, Any], Iterable[Any],
               ml_collections.FieldReference],
    field_reference_replace_map: MutableMapping[int,
                                                ml_collections.FieldReference]
) -> Union[ml_collections.ConfigDict, Mapping[str, Any], Iterable[Any],
           ml_collections.FieldReference]:
  """Returns a copy with replaced field references."""
  if isinstance(obj, ml_collections.FieldReference):
    return field_reference_replace_map.get(id(obj), obj)
  elif isinstance(obj, ml_collections.ConfigDict):
    return type(obj)(
        {
            k: replace_in_config(v, field_reference_replace_map)
            for k, v in obj.items()
        },
        type_safe=obj._type_safe,  # pylint: disable=protected-access
        convert_dict=obj._convert_dict)  # pylint: disable=protected-access
  elif isinstance(obj, Mapping):
    it = ((k, replace_in_config(v, field_reference_replace_map))
          for k, v in obj.items())
    try:
      return type(obj)(it)  # pytype: disable=wrong-arg-count
    except:  # pylint: disable=bare-except
      return dict(it)
  elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
    it = (replace_in_config(v, field_reference_replace_map) for v in obj)
    try:
      return type(obj)(it)  # pytype: disable=wrong-arg-count
    except:  # pylint: disable=bare-except
      return list(it)
  else:
    return obj


def replace_in_func(
    fn: F,
    field_reference_replace_map: MutableMapping[int,
                                                ml_collections.FieldReference],
) -> F:
  """Replace the usages of the field references in the function closure."""
  # We need this function because the module `copy`'s functions don't actually
  # copy the function objects. See https://docs.python.org/3/library/copy.html

  closure_content = replace_in_config(
      (cell.cell_contents for cell in fn.__closure__),
      field_reference_replace_map)
  new_closure = tuple(create_cell_object(x) for x in closure_content)

  new_fn = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__,
                              fn.__defaults__, new_closure)

  # In case the function was given attrs (note this dict is a shallow copy):
  new_fn.__dict__.update(fn.__dict__)

  return new_fn


def replace_usage_of_field_references(
    obj: Union[ml_collections.ConfigDict, Mapping[str, Any], Iterable[Any],
               ml_collections.FieldReference],
    field_reference_replace_map: MutableMapping[int,
                                                ml_collections.FieldReference]
) -> None:
  """Replaces the usages of the field references."""
  if isinstance(obj, ml_collections.FieldReference):
    for i, op in enumerate(obj._ops):  # pylint: disable=protected-access
      fn = replace_in_func(op.fn, field_reference_replace_map)
      new_args = replace_in_config(op.args, field_reference_replace_map)
      obj._ops[i] = ml_collections.config_dict._Op(fn, new_args)  # pylint: disable=protected-access
  else:
    if isinstance(obj, (ml_collections.ConfigDict, Mapping)):
      fields = obj._fields if isinstance(  # pylint: disable=protected-access
          obj, ml_collections.ConfigDict) else obj
      it = fields.values()
    else:
      it = obj

    for v in it:
      if isinstance(v, ml_collections.FieldReference):
        replace_usage_of_field_references(v, field_reference_replace_map)
      elif isinstance(v, (ml_collections.ConfigDict,
                          Iterable)) and not isinstance(obj, (str, bytes)):
        replace_usage_of_field_references(v, field_reference_replace_map)


def safe_deepcopy_config(
    config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Deep-copies a config by also substituting the usage of the field refs."""
  new_config = copy.deepcopy(config)

  field_reference_replace_map = {}
  _compute_field_reference_replace_map(config, new_config,
                                       field_reference_replace_map)

  with new_config.unlocked():
    replace_usage_of_field_references(new_config, field_reference_replace_map)

  return new_config


def update_config_including_refs(config: ml_collections.ConfigDict, *other,
                                 **kwargs) -> None:
  """Like `config.update(*other) but also setting the field references."""
  if len(other) > 1:
    raise TypeError('update expected at most 1 arguments, got {}'.format(
        len(other)))
  for other in other + (kwargs,):
    iteritems_kwargs = {}
    if isinstance(other, ml_collections.ConfigDict):
      iteritems_kwargs['preserve_field_references'] = True

    for key, value in other.items(**iteritems_kwargs):  # pytype: disable=wrong-keyword-args
      if key not in config:
        config[key] = value
      elif isinstance(config._fields[key], ml_collections.ConfigDict):  # pylint: disable=protected-access
        update_config_including_refs(config[key], other[key])
      else:
        config[key] = value


def get_device_memory(platform_type: Optional[str], megacore: bool) -> int:
  """Return the total amount of memory per device."""
  if (first_device := jax.devices()[0]).platform == 'cpu':
    return 16
  elif first_device.platform == 'gpu':
    if platform_type in {'p100', 'v100'}:
      return 16
    elif platform_type == 'p4':
      return 8
    elif platform_type == 'a100':
      return 40
    else:
      return 16
  elif first_device.platform == 'tpu':
    if first_device.device_kind in {'Cloud TPU', 'TPU v2', 'TPU v3'}:
      return 16
    elif first_device.device_kind == 'TPU v4':
      if platform_type == 'pl':
        return 8
      else:
        return 32 if megacore else 16
    else:
      raise ValueError(f'Unrecognized device kind: {first_device.device_kind}')
  else:
    raise ValueError(f'Unrecognized device type: {first_device.platform}')


def create_tokenizer(
    config: ml_collections.ConfigDict) -> tokenizers.TextTokenizer:
  """Creates a DMVR tokenizer."""
  tokenizer_type = config.tokenizer_type
  vocab_path = config.get('tokenizer_vocab', config.get('vocabulary_path'))

  if tokenizer_type == 'bert':
    return tokenizers.BertTokenizer(vocab_path)
  elif tokenizer_type == 'clip':
    return tokenizers.ClipTokenizer()
  elif tokenizer_type == 't5':
    return tokenizers.SentencePieceTokenizer(vocab_path)
  else:
    raise ValueError(f'Unknown tokenizer type {tokenizer_type!r} '
                     f'Supported: "bert", "clip", "t5"')
