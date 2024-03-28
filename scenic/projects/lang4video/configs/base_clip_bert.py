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

"""Base config for CLIP+BERT."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_clip

def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip_clip.get_config(run_local)

  config.experiment_name = 'clip_bert'

  config.model.text_encoder_name = 'bert'
  config.model.text_encoder = ml_collections.ConfigDict()
  config.model.text_encoder.config_name = 'base'

  config.dataset_configs.vocab_size = 30_522
  # TODO(sacastro): try cased. We'd need to change the DMVR BertTokenizer.
  config.dataset_configs.lowercase = True

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.tokenizer.tokenizer_type = 'bert'

  # Add vocab path.
  config.dataset_configs.tokenizer.tokenizer_vocab = ''

  return config
