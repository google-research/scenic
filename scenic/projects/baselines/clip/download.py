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

"""Provides checkpoint download helpers."""

import hashlib
import os
import tempfile
from typing import Optional
import urllib

from absl import logging
from tensorflow.io import gfile
import tqdm

DEFAULT_DOWNLOAD_DIR = os.path.expanduser('~/.cache/scenic/clip')


def hash_file(path):
  return hashlib.sha256(gfile.GFile(path, 'rb').read()).hexdigest()


def download(
    url: str,
    root: str = DEFAULT_DOWNLOAD_DIR,
    expected_sha256: Optional[str] = None
):
  """Download a file if it does not exist, with a progress bar.

  Based on https://github.com/openai/CLIP/blob/main/clip/clip.py#L4

  Args:
    url (str): URL of file to download.
    root (str): Directory to place the downloaded file.
    expected_sha256: Optional sha256 sum. If provided, checks downloaded file.
  Raises:
    RuntimeError: Downloaded file existed as a directory, or sha256 of dowload
                  does not match expected_sha256.
  Returns:
    download_target (str): path to downloaded file
  """
  gfile.makedirs(root)
  filename = os.path.basename(url)
  if '?' in filename:
    # strip trailing HTTP GET arguments
    filename = filename[:filename.rindex('?')]

  download_target = os.path.join(root, filename)

  if gfile.exists(download_target):
    if gfile.isdir(download_target):
      raise RuntimeError(f'{download_target} exists and is not a regular file')
    elif expected_sha256:
      if hash_file(download_target) == expected_sha256:
        return download_target
      logging.warning('%s exists, but the SHA256 checksum does not match;'
                      're-downloading the file', download_target)

  temp_file = tempfile.NamedTemporaryFile(delete=False).name
  with gfile.GFile(temp_file, 'wb') as output:
    with urllib.request.urlopen(url) as source:
      loop = tqdm.tqdm(total=int(source.info().get('Content-Length')),
                       ncols=80, unit='iB', unit_scale=True, unit_divisor=1024)
      while True:
        buffer = source.read(8192)
        if not buffer:
          break

        output.write(buffer)
        loop.update(len(buffer))

  if expected_sha256 and hash_file(temp_file) != expected_sha256:
    raise RuntimeError(
        'Model has been downloaded but the SHA256 checksum does not not match')

  # Use copy+remove instead of rename in case source and target are on different
  # file systems:
  gfile.copy(temp_file, download_target, overwrite=True)
  gfile.remove(temp_file)

  return download_target
