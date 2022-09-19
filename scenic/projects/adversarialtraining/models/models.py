"""Registry for the available models we can train."""

from typing import Type

from scenic.model_lib.base_models import base_model
from scenic.projects.adversarialtraining.models import vit_advtrain

ALL_MODELS = {}

CLASSIFICATION_MODELS = {
    'vit_advtrain_multilabel_classification':
        vit_advtrain.ViTMultiLabelClassificationModel,
}


ALL_MODELS.update(CLASSIFICATION_MODELS)


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


def get_kwargs(config, use_aux_batchnorm, use_aux_dropout):
  """Get parameterization for call."""
  if config.model_name == 'resnet_advtrain_classification':
    apply_kwargs = {
        'use_aux_batchnorm': use_aux_batchnorm,
        'use_aux_dropout': use_aux_dropout,
    }
  elif config.model_name == 'vit_advtrain_multilabel_classification':
    apply_kwargs = {
        'use_aux_dropout': use_aux_dropout,
    }
  else:
    raise ValueError('Unknown model: %s' % config.model_name)
  return apply_kwargs
