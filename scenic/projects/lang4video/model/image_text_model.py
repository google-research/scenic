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

"""Scenic two-tower video and text model."""

from collections.abc import Mapping
from typing import Optional
from typing import TypeVar

import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models import base_model
from scenic.projects.lang4video.model import loss
from scenic.projects.lang4video.model.base_encoders import Encoder
from scenic.projects.lang4video.model.base_encoders import ImageEncoder
from scenic.projects.lang4video.model.base_encoders import ImageTextEncoder
from scenic.projects.lang4video.model.base_encoders import TextEncoder
from scenic.projects.lang4video.model.bert import BertTextEncoder
from scenic.projects.lang4video.model.clip import ClipImageEncoder
from scenic.projects.lang4video.model.clip import ClipTextEncoder
from scenic.projects.lang4video.model.mlp import MlpImageEncoder
from scenic.projects.lang4video.model.mlp import MlpTextEncoder
from scenic.projects.lang4video.model.sequential import SequentialImageEncoder
from scenic.projects.lang4video.model.sequential import SequentialTextEncoder
from scenic.projects.lang4video.model.t5 import T5TextEncoder
from scenic.projects.lang4video.model.transformer import TransformerImageEncoder
from scenic.projects.lang4video.model.transformer import TransformerTextEncoder

T = TypeVar('T', bound=Encoder)

IMAGE_ENCODER_CLASS_MAP: Mapping[str, type[ImageEncoder]] = {
    'clip': ClipImageEncoder,
    'mlp': MlpImageEncoder,
    'sequential': SequentialImageEncoder,
    'transformer': TransformerImageEncoder,
}

TEXT_ENCODER_CLASS_MAP: Mapping[str, type[TextEncoder]] = {
    'bert': BertTextEncoder,
    'clip': ClipTextEncoder,
    'mlp': MlpTextEncoder,
    'sequential': SequentialTextEncoder,
    't5': T5TextEncoder,
    'transformer': TransformerTextEncoder,
}


def _create_encoder(
    config: ml_collections.ConfigDict,
    key_prefix: str,
    class_map: Mapping[str, type[T]],
    **kwargs,
) -> T:
  """Creates an encoder from a config."""
  name = config.get(f'{key_prefix}_encoder_name') or config['encoder_name']
  encoder_config = config.get(f'{key_prefix}_encoder',
                              config.get('encoder', {}))

  encoder_class = class_map[name]

  if name == 'sequential':
    # We use dicts to set the encoders because lists of dicts aren't supported
    # if somebody wants to freeze the config.
    #
    # Note the keys should follow an alphabetical order for the encoder order to
    # be fine. This is because `ConfigDict` orders the keys in this way.
    kwargs['encoders'] = [
        _create_encoder(sub_encoder_config, key_prefix, class_map, **kwargs)
        for sub_encoder_config in encoder_config.get('encoders', {}).values()
    ]
  else:
    kwargs.update(**encoder_config)

  return encoder_class(**kwargs)  # pytype: disable=not-instantiable


def _create_image_encoder(
    config: ml_collections.ConfigDict,
    **kwargs,
) -> ImageEncoder:
  return _create_encoder(
      config, key_prefix='image', class_map=IMAGE_ENCODER_CLASS_MAP, **kwargs)


def _create_text_encoder(
    config: ml_collections.ConfigDict,
    **kwargs,
) -> TextEncoder:
  return _create_encoder(
      config, key_prefix='text', class_map=TEXT_ENCODER_CLASS_MAP, **kwargs)


class ImageTextModel(base_model.BaseModel):
  """Scenic two-tower image-text model."""

  flax_model: ImageTextEncoder

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    return lambda scores, batch: {}

  def loss_function(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      scores: jnp.ndarray,  # Shape: (N, N)
      batch: Optional[base_model.Batch] = None,
      model_params: Optional[jnp.ndarray] = None,
      where: Optional[jnp.ndarray] = None,
      initial: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:  # Shape: (N,)
    loss_fn_name = self.config.model.get('loss', 'nce')
    if loss_fn_name == 'nce':
      return loss.nce_loss(scores, where=where, initial=initial)
    elif loss_fn_name == 'distance':
      return -scores.diagonal()
    else:
      raise ValueError(f'Unrecognized loss function: {loss_fn_name}')

  def build_flax_model(self) -> ImageTextEncoder:
    model_config = self.config.model

    kwargs = {}

    if dtype := self.config.get('model_dtype_str'):
      kwargs['dtype'] = jnp.dtype(dtype)

    return ImageTextEncoder(
        image_encoder=_create_image_encoder(model_config, **kwargs),
        text_encoder=_create_text_encoder(model_config, **kwargs),
        similarity=model_config.get('similarity', 'dot_product'),
        logit_scale=-jnp.log(self.config.get('temperature', 0.01)),
        **kwargs)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict()
