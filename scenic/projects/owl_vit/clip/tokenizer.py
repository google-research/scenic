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
from typing import List, Optional

from clip import simple_tokenizer
from scenic.projects.baselines.clip import download


# pylint: disable=line-too-long
DEFAULT_BPE_PATH = None
DEFAULT_BPE_URL = 'https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true'
# pylint: enable=line-too-long


def tokenize(text: str, max_token_len: int = 77) -> List[int]:
  tokenizer = build_tokenizer()
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']
  tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
  output = [0] * max_token_len
  output[:min(max_token_len, len(tokens))] = tokens[:max_token_len]
  return output


@functools.lru_cache(maxsize=1)
def build_tokenizer(
    bpe_path: Optional[str] = DEFAULT_BPE_PATH,
    bpe_url: str = DEFAULT_BPE_URL,
    download_dir: str = download.DEFAULT_DOWNLOAD_DIR
) -> simple_tokenizer.SimpleTokenizer:
  """Returns CLIP's tokenizer."""
  if bpe_path is None:
    bpe_path = download.download(bpe_url, download_dir)
    logging.info('Downloaded vocabulary from %s to %s', bpe_url, download_dir)

  return simple_tokenizer.SimpleTokenizer(bpe_path)
