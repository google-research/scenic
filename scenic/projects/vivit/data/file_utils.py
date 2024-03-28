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

"""Utility functions for working with sharded files.

A sharded file is a single conceptual file that is broken into a collection
of files to make parallelization easier.  A sharded file spec is like a
filename for a sharded file; the file spec "/some/path/prefix@200.txt"
says that the sharded file consists of 200 actual files that have names like
"/some/path/prefix-00000-of-00200.txt", "/some/path/prefix-00001-of-00200.txt",
etc.  This module contains functions for parsing, generating and detecting
sharded file specs.
"""

import math
import re
from typing import Iterator, Tuple

SHARD_SPEC_PATTERN = re.compile(R'((.*)\@(\d*[1-9]\d*)(?:\.(.+))?)')


class ShardError(Exception):
  """An I/O error."""


def parse_sharded_file_spec(spec: str) -> Tuple[str, int, str]:
  """Parse a sharded file specification.

  Args:
    spec: The sharded file specification. A sharded file spec is one like
      'gs://some/file@200.txt'. Here, '@200' specifies the number of shards.

  Returns:
    basename: The basename for the files.
    num_shards: The number of shards.
    suffix: The suffix if there is one, or '' if not.
  Raises:
    ShardError: If the spec is not a valid sharded specification.
  """
  m = SHARD_SPEC_PATTERN.match(spec)
  if not m:
    raise ShardError(('The file specification {0} is not a sharded file '
                      'specification because it did not match the regex '
                      '{1}').format(spec, SHARD_SPEC_PATTERN.pattern))

  # If there's a non-empty suffix, we need to prepend '.' so we get files like
  # foo@20.ext instead of foo@ext
  suffix = '.' + m.group(4) if m.group(4) else ''

  return m.group(2), int(m.group(3)), suffix


def _shard_width(num_shards: int) -> int:
  """Return the width of the shard matcher based on the number of shards."""
  return max(5, int(math.floor(math.log10(num_shards)) + 1))


def generate_sharded_filenames(spec: str) -> Iterator[str]:
  """Generator for the list of filenames corresponding to the sharding path.

  Args:
    spec: Represents a filename with a sharding specification.
      e.g., 'gs://some/file@200.txt' represents a file sharded 200 ways.

  Yields:
    Each filename in the sharding path.

  Raises:
    ShardError: If spec is not a valid sharded file specification.
  """
  basename, num_shards, suffix = parse_sharded_file_spec(spec)
  width = _shard_width(num_shards)
  format_str = '{{0}}-{{1:0{0}}}-of-{{2:0{0}}}{{3}}'.format(width)
  for i in range(num_shards):
    yield format_str.format(basename, i, num_shards, suffix)


def is_sharded_file_spec(spec: str) -> bool:
  """Returns True if spec is a sharded file specification."""
  m = SHARD_SPEC_PATTERN.match(spec)
  return m is not None
