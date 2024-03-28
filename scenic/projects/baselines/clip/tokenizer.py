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

"""Simple CLIP tokenizer wrapper."""

from absl import logging
import functools
from typing import Any, Callable, Optional, Sequence, Union

from clip.simple_tokenizer import SimpleTokenizer
import numpy as np
from scenic.projects.baselines.clip import download


# pylint: disable=line-too-long
DEFAULT_BPE_PATH = None
DEFAULT_BPE_URL = 'https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true'
MAX_TEXT_LENGTH = 77
# pylint: enable=line-too-long


def _tokenize(texts: Union[str, Sequence[str]], tokenizer: Any,
              context_length: int, truncate: bool = False) -> np.ndarray:
  """Tokenizes texts using tokenizer."""
  if isinstance(texts, str):
    texts = [texts]
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']
  all_tokens = [
      [sot_token] + tokenizer.encode(text) + [eot_token] for text in texts
  ]
  result = np.zeros((len(all_tokens), context_length), dtype=int)
  for i, tokens in enumerate(all_tokens):
    if len(tokens) > context_length:
      if truncate:
        tokens = tokens[:context_length - 1] + [eot_token]
      else:
        raise RuntimeError(
            f'Input {texts[i]} is too long for context length {context_length}')

    result[i, :len(tokens)] = np.asarray(tokens)
  return result


def build_tokenizer(
    bpe_path: Optional[str] = DEFAULT_BPE_PATH,
    truncate: Optional[bool] = False,
    bpe_url: str = DEFAULT_BPE_URL,
    download_dir: str = download.DEFAULT_DOWNLOAD_DIR
) -> Callable[[Union[str, Sequence[str]]], np.ndarray]:
  """Returns CLIP's tokenization function."""
  if bpe_path is None:
    bpe_path = download.download(bpe_url, download_dir)
    logging.info('Downloaded vocabulary from %s to %s', bpe_url, download_dir)

  tokenizer = SimpleTokenizer(bpe_path)
  tokenizer_fn = functools.partial(_tokenize, tokenizer=tokenizer,
                                   context_length=MAX_TEXT_LENGTH,
                                   truncate=truncate)
  return tokenizer_fn
