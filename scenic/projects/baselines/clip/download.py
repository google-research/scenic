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

  try:
    gfile.rename(temp_file, download_target, overwrite=True)
  except:
    gfile.copy(temp_file, download_target, overwrite=True)

  return download_target
