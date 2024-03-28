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

"""Base config for CLIP+T5."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_clip


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip_clip.get_config(run_local)

  config.experiment_name = 'clip_t5'

  config.model.text_encoder_name = 't5'
  config.model.text_encoder = ml_collections.ConfigDict()
  config.model.text_encoder.config_name = 'base'

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.tokenizer.tokenizer_type = 't5'

  config.dataset_configs.tokenizer.tokenizer_vocab = (
      'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')
  config.dataset_configs.tokenizer.prepend_bos = False

  config.dataset_configs.tokenizer_type = 'sentence_piece'
  config.dataset_configs.vocab_size = 32_128

  return config
