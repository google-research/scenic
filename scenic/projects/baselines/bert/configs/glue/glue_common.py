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

"""BERT Common configurations."""

import ml_collections

EMBEDDING_WIDTH = {'Ti': 128, 'S': 128, 'B': 768, 'L': 1024}
HIDDEN_SIZE = {'Ti': 128, 'S': 256, 'B': 768, 'L': 1024}
NUM_HEADS = {'Ti': 2, 'S': 4, 'B': 12, 'L': 16}
MLP_DIM = {'Ti': 512, 'S': 1024, 'B': 3072, 'L': 4096}
NUM_LAYERS = {'Ti': 6, 'S': 12, 'B': 12, 'L': 24}

# Classification data path:
INPUT_MEAT_DATA_PATH = '/path/to/data/tfrecords/{task_path}_meta_data'
TRAIN_DATA_PATH = '/path/to/data/tfrecords/{task_path}_train.tf_record'
EVAL_DATA_PATH = '/path/to/data/tfrecords/{task_path}_eval.tf_record'


# task_name to task_path
GLUE_TASK_PATH = {
    # From glue
    'mnli_matched': 'MNLI/MNLI_matched',
    'mnli_mismatched': 'MNLI/MNLI_mismatched',
    'qqp': 'QQP/QQP',
    'qnli': 'QNLI/QNLI',
    'sst2': 'SST-2/SST-2',
    'cola': 'COLA/COLA',
    'stsb': 'STS-B/STS-B',
    'mrpc': 'MRPC/MRPC',
    'rte': 'RTE/RTE',
    # GLUE webpage notes that there are issues with the construction of WNLI.
    'wnli': 'WNLI/WNLI',
    # AX is the GLUE diagnostics dataset (https://gluebenchmark.com/diagnostics)
    # which provides examples usedful to debug and diagnose models
    'ax': 'AX/AX',
}


def get_config():
  """Dummy get_config function to pass tests."""
  return ml_collections.ConfigDict()
