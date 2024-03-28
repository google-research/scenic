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

"""Utils for the trainers."""

import collections
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
import functools
from typing import Any
from typing import Hashable
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
from flax import traverse_util
from flax.core.frozen_dict import FrozenDict
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.lang4video import util
from scenic.projects.lang4video.model.base_encoders import ImageTextEncoder
from scenic.train_lib import train_utils

F = TypeVar('F', bound=Callable)

NUM_DEVICES_AXIS_NAME = 'num_devices'


def _recursively_merge_frozen_dicts(
    target: FrozenDict,
    source: FrozenDict,
) -> tuple[FrozenDict, Iterable[str], Iterable[str]]:
  """Recursively merges the source dict into the target one."""
  target = traverse_util.flatten_dict(target)
  source = traverse_util.flatten_dict(source)

  missing_keys = []
  for k in target:
    if k in source:
      target[k] = source[k]
    else:
      # Don't use the '/' directly as the `flatten_dict` `sep` argument because
      # there could be params that contain this string (or others).
      missing_keys.append('/'.join(k))

  unexpected_keys = ['/'.join(k) for k in set(source) - set(target)]

  target = flax.core.freeze(traverse_util.unflatten_dict(target))

  return target, missing_keys, unexpected_keys


def init_encoder(
    encoder: ImageTextEncoder,
    input_spec: Sequence[tuple[tuple[int, ...], Any]],
    config: ml_collections.ConfigDict,
    rng: Optional[jnp.ndarray] = None,
    method: Optional[Callable[..., Any]] = None,  # pylint: disable=unused-argument
    strict_for_missing: bool = False,
    strict_for_unexpected: bool = True,
) -> tuple[FrozenDict, FrozenDict]:
  """Creates an instance of the given model class and the train state."""
  rng = jax.random.PRNGKey(0) if rng is None else rng

  params, model_state, _, _ = train_utils.initialize_model(
      model_def=encoder,
      input_spec=input_spec,
      # method=method,
      config=config,
      rngs=rng)

  if config.model.get('load_pretrained_vars'):
    logging.info('Loading the pretrained model…')
    pretrained_params, pretrained_model_state = encoder.get_pretrained_vars()
    logging.info('Pretrained model loaded.')

    msgs = []

    params, missing_param_keys, unexpected_param_keys = _recursively_merge_frozen_dicts(
        params, pretrained_params)

    if missing_param_keys:
      msgs.append(
          f'Missing params in the pretrained model: {missing_param_keys}')

    if unexpected_param_keys:
      msgs.append('Unexpected params in the pretrained model:'
                  f' {unexpected_param_keys}')

    model_state, missing_model_state_keys, unexpected_model_state_keys = _recursively_merge_frozen_dicts(
        model_state, pretrained_model_state)

    if missing_model_state_keys:
      msgs.append('Missing model state keys in the pretrained model:'
                  f' {missing_model_state_keys}')

    if unexpected_model_state_keys:
      msgs.append('Unexpected model state keys in the pretrained model:'
                  f' {unexpected_model_state_keys}')

    if msgs:
      if strict_for_missing and (missing_param_keys or
                                 missing_model_state_keys):
        raise RuntimeError('Error loading the pretrained model:\n' +
                           '\n\t'.join(msgs))
      elif strict_for_unexpected and (unexpected_param_keys or
                                      unexpected_model_state_keys):
        raise RuntimeError('Error loading the pretrained model:\n' +
                           '\n\t'.join(msgs))
      else:
        for msg in msgs:
          logging.warning(msg)

  logging.info(
      'Params dtype counts: %s',
      collections.Counter(
          t.dtype for t in traverse_util.flatten_dict(params).values()))

  logging.info('Encoder: %s', encoder)

  return params, model_state


def get_epoch_steps(
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    split: Literal['train', 'eval', 'test'],
) -> int:
  """Computes the steps per epoch given the config, dataset, and split."""
  # We need to iterate based on the number of steps (as opposed to just
  # iterating the dataset) because Scenic gives an infinite iteration and
  # because the total steps may be overridden by the config.
  num_eval_examples = dataset.meta_data[f'num_{split}_examples']

  if split in {'eval', 'test'}:
    batch_size = config.get('eval_batch_size', config.batch_size)
  else:
    batch_size = config.batch_size

  total_steps = int(jnp.ceil(num_eval_examples / batch_size))

  return config.get(f'steps_per_{split}') or total_steps


def log_eval_summary(
    writer: metric_writers.MetricWriter,
    eval_metrics: Optional[Sequence[Mapping[str, tuple[float, int]]]] = None,
    extra_eval_summary: Optional[Mapping[str, float]] = None,
    step: int = 0,
    prefix: str = 'valid',
    key_separator: str = '/',
) -> Mapping[str, float]:
  return train_utils.log_eval_summary(
      step=step,
      eval_metrics=eval_metrics or [{}],
      extra_eval_summary=extra_eval_summary,
      writer=writer,
      prefix=prefix,
      key_separator=key_separator)


def pmap_maybe(
    fun: F,
    *args,
    pmap_enabled: Optional[bool] = None,
    **kwargs,
) -> F:
  """Returns a pmapped version of `fun` if `pmap_enabled` is true, else `fun`.

  When `fun` is not pmapped, the input args and the output are unreplicated.

  Args:
    fun: see `pmap` docs.
    *args: are passed to `pmap` if it's called.
    pmap_enabled: whether to return a pmapped version of `fun`. By default, it
      doesn't pmap `fun` if there's a single device that's a CPU.
    **kwargs: are passed to `pmap` if it's called.
  """
  pmap_enabled = (jax.device_count() != 1 or jax.devices()[0].platform != 'cpu'
                 ) if pmap_enabled is None else pmap_enabled

  if pmap_enabled:
    logging.info('Doing pmap.')
    return jax.pmap(fun, *args, **kwargs)
  else:
    logging.info('Skipping pmap.')

    # `vmap` can work, however the batch size may be too big for one device.
    # So we just take the first element.
    #
    # And for vmap, passing `axis_name` from `args` makes it fail. It seems to
    # break what `value_and_grad` expects (a scalar).

    def _fun(*args_, **kwargs_):
      args_ = jax_utils.unreplicate(args_)
      kwargs_ = jax_utils.unreplicate(kwargs_)
      return jax_utils.replicate(fun(*args_, **kwargs_))

    return _fun


def axis_name_exists(axis_name: Hashable) -> bool:
  return any(frame.name == axis_name
             for frame in jax.core.thread_local_state.trace_state.axis_env)


# The following are the same functions in `jax.example_libraries.optimizers` but
# `clip_grads` adds an `eps` arg.


def l2_norm(tree: Any) -> jnp.ndarray:
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves, _ = jax.tree_util.tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_grads(
    grad_tree: Any,
    max_norm: float,
    eps: Optional[float] = None,
) -> Any:
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)

  eps = jnp.finfo(norm.dtype).eps if eps is None else eps

  def normalize(g):
    return jnp.where(norm < max_norm, g, g * (max_norm / norm + eps))

  return jax.tree_util.tree_map(normalize, grad_tree)


def get_cached_fn(
    fn: Optional[F] = None,
    *,
    max_size: Optional[int] = 16,
) -> F:
  """Returns a version of `fn` with cache and that logs statistics."""

  def decorator_fn(fn_: F) -> F:
    cached_fn = functools.lru_cache(maxsize=max_size)(fn_)

    @functools.wraps(fn_)
    def wrapper(*args, **kwargs):
      try:
        output = cached_fn(*args, **kwargs)
        logging.info('%s cache stats: %s', fn_, cached_fn.cache_info())
        return output
      except TypeError as e:
        # TODO(sacastro): what if this "unhashable type" error occurs because of
        #   something inside the call to `fn`? Calling the `fn` twice may have
        #   secondary effects. It's unlikely though.
        if ((e_args := getattr(e, 'args', [])) and e_args and
            isinstance(e_args[0], str) and
            e_args[0].startswith('unhashable type: ')):
          logging.warning('Cannot cache %s, %s', fn_, e_args[0])
          return fn_(*args, **kwargs)
        else:
          raise

    return wrapper

  return decorator_fn if fn is None else decorator_fn(fn)


@get_cached_fn
def partial_with_cache(fn: Callable[..., Any], *args,
                       **kwargs) -> Callable[..., Any]:
  return functools.partial(
      fn,
      *args,
      **kwargs,
  )


def decode_image(
    image: jnp.ndarray,
    config: ml_collections.ConfigDict,
) -> np.ndarray:
  """Decodes an image into a BGR image ready to show."""
  return (
      np.asarray(image) * np.array(config.dataset_configs.normalization_std) +
      np.array(config.dataset_configs.normalization_mean))[..., ::-1]


def decode_text(
    text: jnp.ndarray,
    config: ml_collections.ConfigDict,
) -> str:
  """Decodes a tokenized text into a string."""
  tokenizer = util.create_tokenizer(config.dataset_configs.tokenizer)
  tokenizer.initialize()
  return tokenizer.indices_to_string(text.tolist())


def decode_image_and_text(
    image: jnp.ndarray,
    text: jnp.ndarray,
    config: ml_collections.ConfigDict,
) -> tuple[np.ndarray, str]:
  """Decodes an image-text pair into a BGR image ready to show and a string."""
  return decode_image(image, config), decode_text(text, config)


def decode_example_in_batch(
    batch: Mapping[str, jnp.ndarray],
    config: ml_collections.ConfigDict,
    position: tuple[int, ...] = (0, 0),
    frame_index: int = 0,
) -> tuple[np.ndarray, str]:
  """Decodes a batch example into a BGR image ready to show and a string."""
  if (tokenized_text := batch.get('text_indices')) is None:
    tokenized_texts = batch['label']
  else:
    tokenized_texts = tokenized_text[:, :, 0]

  tokenized_text = tokenized_texts[position]

  visual = batch['inputs'][position]

  if visual.ndim == 4:
    visual = visual[frame_index]
  assert visual.ndim == 3

  return decode_image_and_text(visual, tokenized_text, config)


def pad_array_to_be_divisible(
    a: jnp.ndarray,
    divisor: int,
    axis: int = 0,
) -> tuple[jnp.ndarray, int]:
  """Returns `a` padded at the end of `axis` so it's divisible by `divisor`."""
  # This padding size is what we need to sum `n` to get a multiple of
  # `retrieval_batch_size`. Note that the simple remainder gives us what we need
  # to *subtract* from `n`, not sum.
  padding_size = (divisor - (a.shape[axis] % divisor)) % divisor
  padding = jnp.empty((*a.shape[:axis], padding_size, *a.shape[axis + 1:]),
                      dtype=a.dtype)
  return jnp.concatenate((a, padding), axis=axis), padding_size


T = TypeVar('T')


def pad_and_batch(tree: T, batch_size: int, axis: int = 0) -> tuple[T, int]:
  """Pads and batch the pytree."""
  # We peek the first padding size, to then check that the same one applies to
  # all arguments. We can't support multiple padding sizes because we don't
  # know which padding sizes apply to each output argument. So we just use the
  # same one for everything.
  first = jax.tree_util.tree_flatten(tree)[0][0]
  padding_size = pad_array_to_be_divisible(first, batch_size, axis=axis)[1]

  def _pad_and_batch(a: jnp.ndarray) -> jnp.ndarray:
    a, padding_size_ = pad_array_to_be_divisible(a, batch_size, axis=axis)
    assert padding_size_ == padding_size, (
        f'Multiple padding sizes aren\'t '
        f'supported because it\'s hard to know which '
        f'padding sizes will apply to each output '
        f'argument when this inputs are used. '
        f'Actual padding sizes: {(padding_size, padding_size_)}')
    return a.reshape(*a.shape[:axis], -1, batch_size, *a.shape[axis + 1:])

  batched_tree = jax.tree_util.tree_map(_pad_and_batch, tree)

  return batched_tree, padding_size


def unbatch_and_truncate(tree: T, padding_size: int, axis: int = 0) -> T:

  def _unbatch_and_truncate(a: jnp.ndarray) -> jnp.ndarray:
    a = a.reshape(*a.shape[:axis], -1, *a.shape[axis + 2:])
    return a[tuple([slice(None)] * axis +
                   [slice(0, a.shape[axis] - padding_size)])]

  return jax.tree_util.tree_map(_unbatch_and_truncate, tree)


def batch_and_vmap(
    fun: F,
    batch_size: int,
    axis: int = 0,
    axis_name: Optional[Hashable] = None,
    return_padding_size: bool = False,
) -> F:
  """Returns a vmapped function that auto-batches and auto-unbatches."""

  @jax.util.wraps(fun)
  def _batch_and_vmap(tree: Any) -> Union[Any, tuple[Any, int]]:
    batched_tree, padding_size = pad_and_batch(
        tree, batch_size=batch_size, axis=axis)
    output = jax.vmap(
        fun, in_axes=axis, out_axes=axis, axis_name=axis_name)(
            batched_tree)
    output = unbatch_and_truncate(output, padding_size=padding_size)
    return (output, padding_size) if return_padding_size else output

  return _batch_and_vmap


def split_in_batches(
    fun: F,
    batch_size: int,
    axis: int = 0,
    return_padding_size: bool = False,
) -> F:
  """Returns a function that splits an array into batches."""

  @jax.util.wraps(fun)
  def _split_in_batches(tree: Any) -> Union[Any, tuple[Any, int]]:
    batched_tree, padding_size = pad_and_batch(
        tree, batch_size=batch_size, axis=axis)
    output = jax.lax.map(fun, batched_tree)
    output = unbatch_and_truncate(output, padding_size=padding_size)
    return (output, padding_size) if return_padding_size else output

  return _split_in_batches


def compute_mask(
    text: jnp.ndarray,  # Shape: (..., L)
    config: ml_collections.ConfigDict,
) -> Optional[jnp.ndarray]:  # Shape: (..., L)
  # TODO(sacastro): this supposes the padding token ID is 0. It should be more
  #  generic
  del config
  return (text > 0).astype(text.dtype)


ArgSpec = tuple[tuple[int, ...], Any]
InputSpec = tuple[ArgSpec, ArgSpec, ArgSpec]


def get_input_spec(
    dataset_meta_data: Mapping[str, Any],
    dataset_configs: ml_collections.ConfigDict,
    train: bool,
) -> InputSpec:
  """Returns the input spec."""
  if input_spec := (dataset_meta_data.get('input_spec') if train else
                    dataset_meta_data.get('eval_input_spec',
                                          dataset_meta_data.get('input_spec'))):
    visual_spec = input_spec[dataset_configs.visual_key]
    text_spec = input_spec[dataset_configs.text_out_key]
  else:
    visual_spec = (dataset_meta_data['input_shape'],
                   dataset_meta_data.get('input_dtype', jnp.float32))

    text_shape = (
        dataset_meta_data.get('text_shape') or
        dataset_meta_data.get('target_shape') or
        (-1, dataset_configs.get('max_num_words', 32)))
    if len(text_shape) > 2:
      # We skip the dimension for multiple texts per example.
      text_shape = text_shape[:1] + text_shape[2:]

    text_spec = text_shape, dataset_meta_data.get('text_dtype', jnp.int32)

  mask_spec = text_spec

  return visual_spec, text_spec, mask_spec


def is_video_input(input_spec: InputSpec) -> bool:
  return len(input_spec[0][0]) > 4
