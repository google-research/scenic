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
