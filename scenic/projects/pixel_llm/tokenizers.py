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

"""Tokenizer Wrapper."""

from typing import Union

from dmvr import tokenizers as dmvr_tokenizers
from scenic.projects.t5 import tokenizer as t5_tokenizer

BERT_TOKENIZER_PATH = '/path/to/bert_tokenizer/'


class T5Tokenizer(t5_tokenizer.SentencePieceTokenizer):

  @property
  def vocab_size(self) -> int:
    # SP_VOCAB_SIZE
    return self._vocab_size + 28

TOKENIZER = Union[
    dmvr_tokenizers.BertTokenizer,
    # t5_tokenizer.SentencePieceTokenizer,
    T5Tokenizer,
]


def get_tokenizer(tokenizer_weight_path) -> TOKENIZER:
  if tokenizer_weight_path == 't5':
    # tokenizer = t5_tokenizer.build_dmvr_sp_model()
    tokenizer = T5Tokenizer(t5_tokenizer.SP_MODEL_PATH)
  else:
    tokenizer = dmvr_tokenizers.BertTokenizer(tokenizer_weight_path)

  return tokenizer
