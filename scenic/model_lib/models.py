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

"""Registry for the available models we can train."""

from typing import Type

from scenic.model_lib.base_models import base_model
from scenic.projects.baselines import axial_resnet
from scenic.projects.baselines import bit_resnet
from scenic.projects.baselines import fully_connected
from scenic.projects.baselines import hybrid_vit
from scenic.projects.baselines import mixer
from scenic.projects.baselines import resnet
from scenic.projects.baselines import simple_cnn
from scenic.projects.baselines import unet
from scenic.projects.baselines import vit

ALL_MODELS = {}

CLASSIFICATION_MODELS = {
    'fully_connected_classification':
        fully_connected.FullyConnectedClassificationModel,
    'simple_cnn_classification':
        simple_cnn.SimpleCNNClassificationModel,
    'axial_resnet_multilabel_classification':
        axial_resnet.AxialResNetMultiLabelClassificationModel,
    'resnet_classification':
        resnet.ResNetClassificationModel,
    'resnet_multilabel_classification':
        resnet.ResNetMultiLabelClassificationModel,
    'bit_resnet_classification':
        bit_resnet.BitResNetClassificationModel,
    'bit_resnet_multilabel_classification':
        bit_resnet.BitResNetMultiLabelClassificationModel,
    'vit_multilabel_classification':
        vit.ViTMultiLabelClassificationModel,
    'hybrid_vit_multilabel_classification':
        hybrid_vit.HybridViTMultiLabelClassificationModel,
    'mixer_multilabel_classification':
        mixer.MixerMultiLabelClassificationModel,
}

SEGMENTATION_MODELS = {
    'simple_cnn_segmentation': simple_cnn.SimpleCNNSegmentationModel,
    'unet_segmentation': unet.UNetSegmentationModel,
}


ALL_MODELS.update(CLASSIFICATION_MODELS)
ALL_MODELS.update(SEGMENTATION_MODELS)


def get_model_cls(model_name: str) -> Type[base_model.BaseModel]:
  """Get the corresponding model class based on the model string.

  API:
  ```
      model_builder= get_model_cls('fully_connected')
      model = model_builder(config, ...)
  ```

  Args:
    model_name: str; Name of the model, e.g. 'fully_connected'.

  Returns:
    The model architecture (a flax Model) along with its default config.
  Raises:
    ValueError if model_name is unrecognized.
  """
  if model_name not in ALL_MODELS.keys():
    raise ValueError('Unrecognized model: {}'.format(model_name))
  return ALL_MODELS[model_name]
