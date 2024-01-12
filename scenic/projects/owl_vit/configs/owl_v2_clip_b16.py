# Copyright 2023 The Scenic Authors.
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

# pylint: disable=line-too-long
r"""OWL v2 CLIP B/16 config."""
import ml_collections


CHECKPOINTS = {
    # https://arxiv.org/abs/2306.09683 Table 1 row 11:
    'owl2-b16-960-st-ngrams': 'gs://scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams_c7e1b9a',
    # https://arxiv.org/abs/2306.09683 Table 1 row 14:
    'owl2-b16-960-st-ngrams-ft-lvisbase': 'gs://scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-ft-lvisbase_d368398',
    # https://arxiv.org/abs/2306.09683 Figure 5 weight ensemble:
    'owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05': 'gs://scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b',
}

CHECKPOINTS['canonical_checkpoint'] = CHECKPOINTS[
    'owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05'
]


def get_config(init_mode='canonical_checkpoint'):
  """Returns the configuration for text-query-based detection using OWL-ViT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'owl_vit_detection'

  # Dataset.
  config.dataset_name = 'owl_vit'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.input_size = 960
  config.dataset_configs.input_range = None
  config.dataset_configs.max_query_length = 16

  # Model.
  config.model_name = 'text_zero_shot_detection'

  config.model = ml_collections.ConfigDict()
  config.model.normalize = True

  config.model.body = ml_collections.ConfigDict()
  config.model.body.type = 'clip'
  config.model.body.variant = 'vit_b16'
  config.model.body.merge_class_token = 'mul-ln'
  config.model.box_bias = 'both'

  # Objectness head.
  config.model.objectness_head = ml_collections.ConfigDict()
  config.model.objectness_head.stop_gradient = True

  # Init.
  config.init_from = ml_collections.ConfigDict()
  checkpoint_path = CHECKPOINTS.get(init_mode, None)
  if checkpoint_path is None:
    raise ValueError('Unknown init_mode: {}'.format(init_mode))
  config.init_from.checkpoint_path = checkpoint_path

  return config
