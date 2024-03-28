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

"""Constructor functions for the pretrained SentencePiece tokenizer.

This module provides constructor functions for creating the pretrained
SentencePiece tokenizer.

The current DMVR SentencePiece tokenizer always sets `prepend_bos` when
initializing the tensorflow processor and returns a sliced tensor when called
with `prepend_bos=False`. This is problematic when the sentencepiece model is
not trained with the BOS token (predefined and hard-coded as '<S>'), which is
the case for T5 tokenizer. This module contains a wrapper for the DMVR
SentencePiece tokenizer to initialize the tensorflow processor without
prepending BOS. Instead, it prepends a custom BOS token given as an argument.
"""

from collections.abc import Sequence
from typing import Optional
from typing import Union

from dmvr import tokenizers
import tensorflow as tf
import tensorflow_text

# pylint: disable=line-too-long
SP_MODEL_PATH = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
# pylint: enable=line-too-long


class SentencePieceTokenizer(tokenizers.SentencePieceTokenizer):
  """Wrapper around `SentencePieceTokenizer` to keep backwards compatibility.

  The current DMVR SentencePiece tokenizer always sets `prepend_bos` when
  initializing the tensorflow processor and returns a sliced tensor when called
  with `prepend_bos=False`. This is problematic when the sentencepiece model is
  not trained with the BOS token (predefined and hard-coded as '<S>'), which is
  the case for T5 tokenizer. This module contains a wrapper for the DMVR
  SentencePiece tokenizer to initialize the tensorflow processor without
  prepending BOS. Instead, it prepends a custom BOS token given as an argument.
  """

  def __init__(self,
               model_path: str,
               bos_id: int = 0):
    self.bos_id = bos_id
    super().__init__(model_path)

  def initialize(self):
    with tf.io.gfile.GFile(self._model_path, 'rb') as f:
      self._tf_sp_model = tensorflow_text.SentencepieceTokenizer(
          model=f.read(),
          out_type=tf.int32,
          add_bos=False,
          add_eos=True)

  def string_tensor_to_indices(self,
                               string_tensor: Union[tf.Tensor, Sequence[str]],
                               prepend_bos: bool = False,
                               append_eos: bool = False,
                               max_num_tokens: Optional[int] = 32) -> tf.Tensor:
    if self._tf_sp_model is None:
      raise RuntimeError('Model was not initialized. Call `initialize` method.')

    tokenized = self._tf_sp_model.tokenize(string_tensor)
    tokenized = tokenized if append_eos else tokenized[..., :-1]

    # Pad to `max_num_tokens`.
    shape = None if max_num_tokens is None else [None, max_num_tokens]
    tokenized = tokenized.to_tensor(default_value=self._pad_token, shape=shape)

    if prepend_bos:
      tokenized = tf.concat([
          tf.zeros_like(tokenized[..., 0:1]) + self.bos_id, tokenized[..., :-1]
      ], -1)
    return tokenized


def build_dmvr_sp_model(model_path: str = SP_MODEL_PATH):
  return SentencePieceTokenizer(model_path)
