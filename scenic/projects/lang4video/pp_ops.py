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

"""Pre-processing ops."""

from collections.abc import Callable
import functools
from typing import Optional

from dmvr import processors
from dmvr import tokenizers as dmvr_tokenizers
import ml_collections
from scenic.dataset_lib.big_transfer.preprocessing import utils
from scenic.dataset_lib.big_transfer.registry import Registry
from scenic.projects.lang4video import util
import tensorflow as tf


def _create_dmvr_tokenizer(
    tokenizer_type: str,
    vocab_path: Optional[str] = None,
) -> dmvr_tokenizers.TextTokenizer:
  """Creates and initializes a DMVR tokenizer."""
  tokenizer = util.create_tokenizer(
      ml_collections.ConfigDict({
          'tokenizer_type': tokenizer_type,
          'tokenizer_vocab': vocab_path,
      }))
  tokenizer.initialize()
  return tokenizer


@Registry.register('preprocess_ops.dmvr_tokenize', 'function')
@utils.InKeyOutKey(indefault=None, outdefault='labels')
def get_pp_dmvr_tokenize(
    tokenizer_type: str,
    vocab_path: Optional[str] = None,
    prepend_bos: bool = False,
    append_eos: bool = False,
    max_num_tokens: Optional[int] = 77,
) -> Callable[[tf.Tensor], tf.Tensor]:
  """PP op to tokenize a text with a DMVR tokenizer."""
  tokenizer = _create_dmvr_tokenizer(tokenizer_type, vocab_path)

  def _dmvr_tokenize_inner(
      text: tf.Tensor,  # Shape: () or (1,)
  ) -> tf.Tensor:  # Shape: (L,)
    if tf.rank(text) == 0:
      text = tf.reshape(text, (-1,))

    token_ids = tokenizer.string_tensor_to_indices(  # Shape: (1, L)
        text,
        prepend_bos=prepend_bos,
        append_eos=append_eos,
        max_num_tokens=max_num_tokens)
    return tf.squeeze(token_ids, axis=0)

  return _dmvr_tokenize_inner


@Registry.register('preprocess_ops.first', 'function')
@utils.InKeyOutKey(indefault='labels', outdefault='labels')
def get_pp_first() -> Callable[[tf.Tensor], tf.Tensor]:
  """PP op to select the first text out of many."""

  def _first(
      sequence: tf.Tensor,  # Any shape
  ) -> tf.Tensor:  # Shape: ()
    return tf.reshape(sequence, (-1,))[0]

  return _first


@Registry.register('preprocess_ops.random_sample', 'function')
@utils.InKeyOutKey(indefault='labels', outdefault='labels')
def get_pp_random_sample(
    seed: Optional[int] = None,
) -> Callable[[tf.Tensor], tf.Tensor]:
  """PP op to select the first text out of many."""

  def _random_sample(
      sequence: tf.Tensor,  # Any shape
  ) -> tf.Tensor:  # Shape: ()
    sequence = tf.reshape(sequence, (-1,))
    shape = tf.shape(sequence)
    index = tf.random.uniform((), maxval=shape[0], dtype=tf.int32, seed=seed)
    return sequence[index]

  return _random_sample


@Registry.register('preprocess_ops.resize_smallest', 'function')
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_pp_resize_smallest(
    min_resize: int,
    method: str = tf.image.ResizeMethod.BILINEAR,
) -> Callable[[tf.Tensor], tf.Tensor]:
  """PP op to resize an image to the smallest size as in DMVR."""

  def _resize(
      image: tf.Tensor,  # Shape: (..., H, W, C)
  ) -> tf.Tensor:
    shape = tf.shape(input=image)
    input_h = shape[-3]
    input_w = shape[-2]

    output_h = tf.maximum(min_resize, (input_h * min_resize) // input_w)
    output_w = tf.maximum(min_resize, (input_w * min_resize) // input_h)

    def resize_fn() -> tf.Tensor:
      frames_resized = tf.image.resize(
          image, (output_h, output_w), method=method)
      return tf.cast(frames_resized, image.dtype)

    should_resize = tf.math.logical_or(
        tf.not_equal(input_w, output_w), tf.not_equal(input_h, output_h))
    image = tf.cond(
        pred=should_resize, true_fn=resize_fn, false_fn=lambda: image)

    return image

  return _resize


@Registry.register('preprocess_ops.decode_frames', 'function')
@utils.InKeyOutKey()
def get_pp_decode_frames(channels: int = 3) -> Callable[[tf.Tensor], tf.Tensor]:
  """Decode encoded frame image strings, see tf.io.decode_image."""

  def _decode_frames(frame_strings: tf.Tensor) -> tf.Tensor:  # pylint: disable=missing-docstring
    # tf.io.decode_image does not set the shape correctly, so we use
    # tf.io.decode_jpeg, which also works for png, see
    # https://github.com/tensorflow/tensorflow/issues/8551
    return tf.map_fn(
        functools.partial(tf.io.decode_jpeg, channels=channels),
        frame_strings,
        back_prop=False,
        dtype=tf.uint8)

  return _decode_frames


@Registry.register('preprocess_ops.sample_sequence', 'function')
@utils.InKeyOutKey()
def get_pp_sample_sequence(
    num_steps: int,
    random: bool,
    stride: int = 1,
) -> Callable[[tf.Tensor], tf.Tensor]:
  return functools.partial(
      processors.sample_sequence,
      num_steps=num_steps,
      random=random,
      stride=stride)


@Registry.register('preprocess_ops.sample_linspace_sequence', 'function')
@utils.InKeyOutKey()
def ge_pp_sample_linspace_sequence(
    num_windows: int,
    num_steps: int,
    stride: int = 1,
) -> Callable[[tf.Tensor], tf.Tensor]:
  return functools.partial(
      processors.sample_linspace_sequence,
      num_windows=num_windows,
      num_steps=num_steps,
      stride=stride)
