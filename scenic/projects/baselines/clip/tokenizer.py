"""Simple CLIP tokenizer wrapper."""

import functools
from typing import Any, Callable, Union, Sequence

from clip.simple_tokenizer import SimpleTokenizer
import jax.numpy as jnp
import numpy as np


DEFAULT_BPE_PATH = <PATH TO bpe_simple_vocab_16e6.txt.gz FILE FROM
https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz>
MAX_TEXT_LENGTH = 77


def _tokenize(texts: Union[str, Sequence[str]], tokenizer: Any,
              context_length: int) -> jnp.ndarray:
  """Tokenizes texts using tokenizer."""
  if isinstance(texts, str):
    texts = [texts]
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']
  all_tokens = [
      [sot_token] + tokenizer.encode(text) + [eot_token] for text in texts
  ]
  result = np.zeros((len(all_tokens), context_length), dtype=np.long)
  for i, tokens in enumerate(all_tokens):
    if len(tokens) > context_length:
      raise RuntimeError(
          f'Input {texts[i]} is too long for context length {context_length}')
    result[i, :len(tokens)] = np.asarray(tokens)
  return jnp.asarray(result)


def build_tokenizer(
    bpe_path: str = DEFAULT_BPE_PATH
    ) -> Callable[[Union[str, Sequence[str]]], np.ndarray]:
  """Returns CLIP's tokenization function."""
  tokenizer = SimpleTokenizer(bpe_path)
  tokenizer_fn = functools.partial(_tokenize, tokenizer=tokenizer,
                                   context_length=MAX_TEXT_LENGTH)
  return tokenizer_fn
