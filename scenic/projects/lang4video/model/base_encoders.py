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

"""Base image and text encoders."""

import abc
from typing import Literal
from typing import Optional

import flax
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp

# Note: we actually are posing the encoders as sequence-to-vector encoders,
# but sometimes they are sequence-to-sequence and sometimes vector-to-vector.
# TODO(sacastro): clarify/improve this.


class Encoder(nn.Module, abc.ABC):
  """A general encoder."""

  config_name: Optional[str] = None
  dtype: jnp.dtype = jnp.float32

  @abc.abstractmethod
  def __call__(
      self,
      *args,
      # TODO(sacastro): add it but make it match subclass signatures.
      # train: bool = False,
      # debug: bool = False,
      **kwargs,
  ) -> jnp.ndarray:
    raise NotImplementedError

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    """Returns the params and model state for a pretrained model."""
    return flax.core.freeze({}), flax.core.freeze({})


class ImageEncoder(Encoder, abc.ABC):
  """Image encoder."""

  @abc.abstractmethod
  def __call__(
      self,
      image: jnp.ndarray,  # Shape: (N, H, W, C)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # Shape: (N, E)
    raise NotImplementedError

  def encode_video(
      self,
      video: jnp.ndarray,  # Shape: (N, F, H, W, C)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # Shape: (N, E)
    images = video.reshape(-1, *video.shape[-3:])
    encoded_images = self(images, train=train, debug=debug)
    encoded_video = encoded_images.reshape(*video.shape[:-3], -1)
    # Averaging representations is the same as averaging predictions:
    # <t, (i1+i2)/2> = 1/2 <t, i1+i2> = (<t, i1> + <t, i2>) / 2
    # (for dot product).
    return encoded_video.mean(axis=-2)


class TextEncoder(Encoder, abc.ABC):
  """Text encoder."""

  @abc.abstractmethod
  def __call__(
      self,
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # shape: (N, E)
    raise NotImplementedError


class ImageTextEncoder(Encoder, abc.ABC):
  """A two-tower image-text encoder model."""

  # `None` because dataclass inheritance in Python < 3.10 can't have required
  # args in the subclass if the superclass has args with defaults.
  image_encoder: Optional[ImageEncoder] = None
  text_encoder: Optional[TextEncoder] = None

  similarity: Literal['cosine', 'dot_product'] = 'dot_product'
  logit_scale: jnp.ndarray = 0  # pytype: disable=annotation-type-mismatch  # jax-ndarray

  def __call__(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      image: jnp.ndarray,  # Shape: (N, H, W, C)
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:  # Shapes: (N, E) and (N, E)
    return (
        self.image_encoder(image, train=train, debug=debug),  # pylint: disable=not-callable
        self.text_encoder(text, mask, train=train, debug=debug))  # pylint: disable=not-callable

  def encode_image(
      self,
      image: jnp.ndarray,  # Shape: (N, H, W, C)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # Shape: (N, E)
    return self.image_encoder(image, train=train, debug=debug)  # pylint: disable=not-callable

  def encode_text(
      self,
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # shape: (N, E)
    return self.text_encoder(text, mask, train=train, debug=debug)  # pylint: disable=not-callable

  def encode_video(
      self,
      video: jnp.ndarray,  # Shape: (N, F, H, W, C)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # Shape: (N, E)
    return self.image_encoder.encode_video(video, train=train, debug=debug)

  def encode_video_and_text(
      self,
      video: jnp.ndarray,  # Shape: (N, F, H, W, C)
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:  # Shapes: (N, E) and (N, E)
    return (self.image_encoder.encode_video(video, train=train, debug=debug),
            self.text_encoder(text, mask, train=train, debug=debug))  # pylint: disable=not-callable

  def encode_image_and_text_and_do_pretraining(
      self,
      image: jnp.ndarray,  # Shape: (N, H, W, C)
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray],  # Shape: (N, L)
      text_for_mlm: jnp.ndarray,
      segment_ids_for_mlm: jnp.ndarray,
      mask_for_mlm: Optional[jnp.ndarray],
      masked_lm_positions: jnp.ndarray,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # pylint: disable=not-callable
    encoded_image = self.image_encoder(image, train=train, debug=debug)
    encoded_text, mlm_logits = self.text_encoder(  # pytype: disable=wrong-keyword-args
        text,
        mask,
        text_for_mlm=text_for_mlm,
        segment_ids_for_mlm=segment_ids_for_mlm,
        mask_for_mlm=mask_for_mlm,
        masked_lm_positions=masked_lm_positions,
        train=train,
        debug=debug)
    # pylint: enable=not-callable
    return encoded_image, encoded_text, mlm_logits

  def encode_video_and_text_and_do_pretraining(
      self,
      video: jnp.ndarray,  # Shape: (N, F, H, W, C)
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray],  # Shape: (N, L)
      text_for_mlm: jnp.ndarray,
      segment_ids_for_mlm: jnp.ndarray,
      mask_for_mlm: Optional[jnp.ndarray],
      masked_lm_positions: jnp.ndarray,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # pylint: disable=not-callable
    encoded_video = self.image_encoder.encode_video(
        video, train=train, debug=debug)
    encoded_text, mlm_logits = self.text_encoder(   # pytype: disable=wrong-keyword-args
        text,
        mask,
        text_for_mlm=text_for_mlm,
        segment_ids_for_mlm=segment_ids_for_mlm,
        mask_for_mlm=mask_for_mlm,
        masked_lm_positions=masked_lm_positions,
        train=train,
        debug=debug)
    # pylint: enable=not-callable
    return encoded_video, encoded_text, mlm_logits

  def compute_similarity(
      self,
      a: jnp.ndarray,
      b: jnp.ndarray,
      all_gather_axis_name: Optional[str] = None,
  ) -> jnp.ndarray:
    """Computes the similarity score matrix from `a` to `b`."""
    assert self.similarity in {'cosine', 'dot_product'}

    if self.similarity == 'cosine':
      a /= jnp.linalg.norm(a, axis=-1, keepdims=True) + jnp.finfo(a.dtype).eps
      b /= jnp.linalg.norm(b, axis=-1, keepdims=True) + jnp.finfo(b.dtype).eps

    if all_gather_axis_name:
      a = jax.lax.all_gather(a, all_gather_axis_name).reshape(-1, a.shape[-1])
      b = jax.lax.all_gather(b, all_gather_axis_name).reshape(-1, b.shape[-1])

    return jnp.exp(self.logit_scale) * a @ b.T

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    image_params, image_model_state = self.image_encoder.get_pretrained_vars()
    text_params, text_model_state = self.text_encoder.get_pretrained_vars()

    params = flax.core.freeze({
        'image_encoder': image_params,
        'text_encoder': text_params,
    })

    model_state = {}

    if image_batch_stats := image_model_state.get('batch_stats'):
      model_state.setdefault('batch_stats', {})
      model_state['batch_stats']['image_encoder'] = image_batch_stats

    if text_batch_stats := text_model_state.get('batch_stats'):
      model_state.setdefault('batch_stats', {})
      model_state['batch_stats']['text_encoder'] = text_batch_stats

    model_state = flax.core.freeze(model_state)

    return params, model_state
