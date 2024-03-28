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

r"""Prepare the ALD codes for all the entity names."""
from absl import app
from absl import flags
import numpy as np

from tensorflow.io import gfile

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_code_length', 4, 'Number of ALD code tokens.')
flags.DEFINE_string('entity_name_tokens',
                    'path_to_file_with_pretokenized_names',
                    'File containing pre-tokenized entity names.')
flags.DEFINE_string('save_output_path',
                    'path_fo_save_the_ald_codes',
                    'Where to save the tokenized entity names.')


def main(_):
  # From BERT tokenizer.
  eos_token = 102
  vocab_size = 30522

  with gfile.Open(FLAGS.entity_name_tokens, 'rb') as f:
    entity_name_tokens = np.load(f)

  uniques, counts = np.unique(entity_name_tokens, return_counts=True)
  tok2count = np.ones(vocab_size) * (max(counts) + 1)
  # Token value to how many times this token is used in the tokenized entity
  # names.
  tok2count[uniques + 2] = counts

  n_ent = 6084491
  clen = FLAGS.max_code_length
  codes = np.ones((n_ent, clen - 1), dtype=np.int32) * (eos_token - 2)
  extra_token_next = (eos_token - 2) * np.ones((
      n_ent, entity_name_tokens.shape[-1] - clen + 1), dtype=np.int32)

  # Rarest tokens appear first.
  for i in range(n_ent):
    tokens_ids_to_keep = np.argsort(tok2count[entity_name_tokens[i] + 2])
    codes[i] = entity_name_tokens[i][tokens_ids_to_keep[:clen - 1]]
    extra_token_next[i] = entity_name_tokens[i][tokens_ids_to_keep[clen - 1:]]

  ald_codes = np.concatenate(
      [codes, np.ones((n_ent, 1), dtype=np.int32) * (eos_token - 2)], axis=-1)

  set_of_codes = {}
  w_random_token = 0
  indexes = np.arange(n_ent)
  np.random.shuffle(indexes)

  for j, i in enumerate(indexes):
    code = ald_codes[i]
    if j % 1000000 == 0:
      print('Processing ' + str(j) + '/' + str(n_ent))
    code_str = '-'.join([str(int(c))for c in code])
    if code_str not in set_of_codes:
      # This is a new code, we leave it as is.
      set_of_codes[code_str] = i
    else:
      # This code already exists.
      # last valid index for the extra tokens
      last_valid_extra_token = np.where(
          extra_token_next[i] != (eos_token - 2))[0]
      if len(last_valid_extra_token):  # pylint: disable=g-explicit-length-test
        last_valid_extra_token = last_valid_extra_token[-1]
      else:
        last_valid_extra_token = -1
      # position
      last_valid_code = np.where(ald_codes[i] != (eos_token - 2))[0][-1]
      if last_valid_code + 1 >= ald_codes.shape[1]:  # the sequence is full!
        last_valid_code = -2  # we use the last position
      nki = 0
      while code_str in set_of_codes and nki <= last_valid_extra_token:
        ald_codes[i, last_valid_code + 1] = extra_token_next[i, nki]
        code_str = '-'.join([str(int(c))for c in ald_codes[i]])
        nki += 1

      n_trials = 0
      if code_str in set_of_codes:
        w_random_token += 1
        while code_str in set_of_codes and n_trials < 3:
          random_token = ((eos_token - 2) + np.random.choice(
              vocab_size - 2 - (eos_token - 2), size=1))[0]
          ald_codes[i, last_valid_code + 1] = random_token
          code_str = '-'.join([str(int(c)) for c in ald_codes[i]])
          n_trials += 1
      set_of_codes[code_str] = i

  print(w_random_token / n_ent * 100.)
  with gfile.Open(FLAGS.save_output_path, 'wb') as f:
    np.save(f, ald_codes)


if __name__ == '__main__':
  app.run(main)
