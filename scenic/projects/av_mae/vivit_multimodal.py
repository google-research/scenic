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

"""ViViT model for multimodal masked pretraining."""

import functools
from typing import Any, Dict, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models import base_model as base_model_lib
from scenic.model_lib.base_models import model_utils as model_utils_lib
from scenic.model_lib.layers import nn_layers

from scenic.projects.av_mae import base_model
from scenic.projects.av_mae import model_utils
from scenic.projects.mbt import model as mbt_model
from scenic.projects.vivit import model as vivit_model
from scenic.projects.vivit import model_utils as vivit_model_utils


ArrayDict = Dict[str, jnp.ndarray]

# pylint: disable=protected-access
_MBT_CLASSIFICATION_METRICS = mbt_model._MBT_CLASSIFICATION_METRICS
# pylint: enable=protected-access


class ViViTMultiMaskedAutoencoder(nn.Module):
  """ViViT model for Multi-Modality Masked AutoEncoder.


  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_classes_dict: Dictionary with the number of output classes.
    num_heads: Number of self-attention heads.
    num_layers: Number of layers.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    token_mask_probability_dict: Probability of dropping out the input tokens
    during training.
    masking_strategy: Masking strategy used to mask the tokens.
    temporal_encoding_config: ConfigDict which defines the type of input
      encoding when tokenising the video.
    decoder_config: ConfigDict which define the decoder.
    attention_config: ConfigDict which defines the type of spatio-temporal
      attention applied in the model.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer. Linearly
      increases from 0 to the provided value..
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    encoder_strategy: Specify how to combine the modalities in the encoder.
      Choose from: separate_encoders, concat_and_encode, same_encoder.
    decoder_strategy: Specify how to combine the modalities in the decoder,
      Choose from: separate_decoders, same_decoder.
    use_inpainting: Whether or not to use the modality inpaiting strategy.
    normalise_encoder_output: If true, layer normalisation is applied to the
      output of the transformer encoder.
    use_modality_tokens: If True, modality learnable tokens are added.
    fusion_layers: When the encoder strategy is 'encode_and_concat',
      this specify how many layers to use for concatenation.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  num_classes_dict: Dict[str, int]
  patches: ml_collections.ConfigDict
  hidden_size: int
  token_mask_probability_dict: Dict[str, float]
  masking_strategy: str
  temporal_encoding_config: ml_collections.ConfigDict
  decoder_config: ml_collections.ConfigDict
  attention_config: ml_collections.ConfigDict
  dropout_rate: float = 0.
  attention_dropout_rate: float = 0.
  stochastic_droplayer_rate: float = 0.
  classifier: str = 'gap'
  positional_embedding: str = 'sinusoidal_1d'
  positional_embedding_decoder: str = 'sinusoidal_1d'
  encoder_strategy: str = 'concat_and_encode'
  decoder_strategy: str = 'same_decoder'
  use_inpainting: bool = False
  shuffle_inpainted_tokens: bool = False
  normalise_encoder_output: bool = True
  use_modality_tokens: bool = False
  fusion_layers: Optional[int] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: ArrayDict, *, train: bool, debug: bool = False):
    del debug
    # inputs will be a dictionary

    x_tokens_dict = tokenize_input(
        inputs, temporal_encoding_config=self.temporal_encoding_config,
        patches=self.patches, hidden_size=self.hidden_size)

    x_tokens_dict = add_positional_embeddings(
        x_tokens_dict, positional_embedding=self.positional_embedding)

    x_tokens_dict = self.add_modality_token(x_tokens_dict)

    (x_unmasked_dict, token_mask_dict,
     unmasked_indices_dict, masked_indices_dict) = self.mask_tokens(
         x_tokens_dict, train=train)
    # If we want to add a class token, add it here.
    # Note that in MAE, positional encodings are not added to the CLS token.
    x_unmasked_dict = add_cls_token(x_unmasked_dict, self.classifier, None)

    x_unmasked_encoded_dict = apply_encoder(
        x_unmasked_dict,
        encoder_strategy=self.encoder_strategy,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
        normalise_encoder_output=self.normalise_encoder_output,
        fusion_layers=self.fusion_layers,
        train=train)

    if not train:
      return x_unmasked_encoded_dict, token_mask_dict

    # Process entire sequence with the decoder.
    mask_token_dict = {}
    for key  in x_unmasked_encoded_dict:
      mask_token_ = self.param(f'mask_token_{key}',
                               nn.initializers.zeros,
                               (1, 1, self.decoder_config.hidden_size))
      mask_token_dict[key] = mask_token_

    x_unmasked_proj_dict = self.apply_decoder_projection(
        x_unmasked_encoded_dict)

    cls_encoded_dict, x_unmasked_proj_dict = self.remove_cls_token(
        x_unmasked_proj_dict)

    if self.use_inpainting:
      x_all_dict, token_mask_dict = self.unshuffle_tokens_inpainting(
          x_unmasked_proj_dict, x_tokens_dict, mask_token_dict)
    else:
      x_all_dict = self.unshuffle_tokens(x_unmasked_proj_dict,
                                         unmasked_indices_dict,
                                         masked_indices_dict,
                                         x_tokens_dict, mask_token_dict)

    # Note. VideoMAE (Facebook) adds positional encodinggs to the CLS token at
    # the encoder as well. VideoMAE (Tong et al) don't use a CLS token.
    # This implementation does not add positional embeddings to the CLS token,
    # as in the original image MAE of He et al.
    x_all_dict = add_positional_embeddings(
        x_all_dict, positional_embedding=self.positional_embedding_decoder)
    x_all_dict = self.add_modality_token(x_all_dict, name='Decoder')

    # Shuffle the tokens and token_mask accordingly if use_inpainting is True.
    if self.use_inpainting and self.shuffle_inpainted_tokens:
      x_all_dict, token_mask_dict = self.shuffle_tokens_and_token_mask(
          x_all_dict, token_mask_dict, rng=self.make_rng('dropout'))

    x_all_dict = add_cls_token(x_all_dict, self.classifier, cls_encoded_dict)

    x_decoded_dict = self.apply_decoder(x_all_dict, train=train)

    _, x_decoded_dict = self.remove_cls_token(x_decoded_dict)

    x_prelogits_dict = {}
    for key, x_decoded in x_decoded_dict.items():
      x_prelogits = nn_layers.IdentityLayer(name=f'pre_logits_{key}')(x_decoded)
      x_prelogits_dict[key] = x_prelogits

    x_logits_dict = self.apply_dense_layer(x_prelogits_dict)

    return x_logits_dict, token_mask_dict

  def add_modality_token(self, x_tokens_dict: ArrayDict, name: str = 'Encoder'
                         ) -> ArrayDict:
    """Add modality learned tokens."""
    if not self.use_modality_tokens:
      return x_tokens_dict
    for key, x_tokens in x_tokens_dict.items():
      modality_token = self.param(f'{name}_modality_token_{key}',
                                  nn.initializers.zeros,
                                  (1, 1, x_tokens.shape[-1]))
      x_tokens = x_tokens + modality_token
      x_tokens_dict[key] = x_tokens

    return x_tokens_dict

  def apply_dense_layer(self, x_prelogits_dict: ArrayDict) -> ArrayDict:
    """Apply the regressor for each modality."""
    x_logits_dict = {}

    for key, x_prelogits in x_prelogits_dict.items():
      x_logits = nn.Dense(self.num_classes_dict[key],
                          kernel_init=nn.initializers.zeros,
                          name=f'output_projection_{key}')(x_prelogits)
      x_logits_dict[key] = x_logits
    return x_logits_dict

  def apply_decoder(self, x_all_dict: ArrayDict, train: bool) -> ArrayDict:
    """Apply the decoder for each modality."""

    def concat_and_decode():
      x_decoded_dict = {}
      decoder = vivit_model.Encoder(
          temporal_dims=None,
          mlp_dim=self.decoder_config.mlp_dim,
          num_layers=self.decoder_config.num_layers,
          num_heads=self.decoder_config.num_heads,
          attention_config=self.decoder_config.attention_config,
          dropout_rate=self.decoder_config.dropout_rate,
          attention_dropout_rate=self.decoder_config.attention_dropout_rate,
          stochastic_droplayer_rate=(
              self.decoder_config.stochastic_droplayer_rate),
          dtype=self.dtype,
          positional_embedding='none',  # Has already been added.
          normalise_output=self.normalise_encoder_output,
          name='Decoder')

      x_all_input_list = []
      x_all_key_list = []
      for key, x_all in x_all_dict.items():
        x_all_input_list.append(x_all)
        x_all_key_list.append(key)

      x_all_concat_input = jnp.concatenate(x_all_input_list, axis=1)
      x_all_decoded_concat = decoder(x_all_concat_input, train=train)
      start_index = 0
      for key, x_all_ in zip(x_all_key_list, x_all_input_list):
        end_index = start_index + x_all_.shape[1]
        x_decoded_dict[key] = x_all_decoded_concat[:, start_index:end_index]
        start_index = end_index
        assert x_decoded_dict[key].shape[1] == x_all_.shape[1]

      return x_decoded_dict

    def same_decoder():
      x_decoded_dict = {}
      decoder = vivit_model.Encoder(
          temporal_dims=None,
          mlp_dim=self.decoder_config.mlp_dim,
          num_layers=self.decoder_config.num_layers,
          num_heads=self.decoder_config.num_heads,
          attention_config=self.decoder_config.attention_config,
          dropout_rate=self.decoder_config.dropout_rate,
          attention_dropout_rate=self.decoder_config.attention_dropout_rate,
          stochastic_droplayer_rate=(
              self.decoder_config.stochastic_droplayer_rate),
          dtype=self.dtype,
          positional_embedding='none',  # Has already been added.
          normalise_output=self.normalise_encoder_output,
          name='Decoder')

      for key, x_all in x_all_dict.items():
        x_decoded = decoder(x_all, train=train)
        x_decoded_dict[key] = x_decoded
      return x_decoded_dict

    def separate_decoders():
      x_decoded_dict = {}
      for key, x_all in x_all_dict.items():
        x_decoded = vivit_model.Encoder(
            temporal_dims=None,
            mlp_dim=self.decoder_config.mlp_dim,
            num_layers=self.decoder_config.num_layers,
            num_heads=self.decoder_config.num_heads,
            attention_config=self.decoder_config.attention_config,
            dropout_rate=self.decoder_config.dropout_rate,
            attention_dropout_rate=self.decoder_config.attention_dropout_rate,
            stochastic_droplayer_rate=(
                self.decoder_config.stochastic_droplayer_rate),
            dtype=self.dtype,
            positional_embedding='none',  # Has already been added.
            normalise_output=self.normalise_encoder_output,
            name=f'Decoder_{key}')(x_all, train=train)
        x_decoded_dict[key] = x_decoded
      return x_decoded_dict

    if self.decoder_strategy == 'separate_decoders':
      return separate_decoders()
    elif self.decoder_strategy == 'same_decoder':
      return same_decoder()
    elif self.decoder_strategy == 'concat_and_decode':
      return concat_and_decode()
    else:
      raise ValueError(
          f'The decoder strategy {self.decoder_strategy} is not supported!')

  def unshuffle_tokens_inpainting(self,
                                  x_unmasked_proj_dict: ArrayDict,
                                  x_tokens_dict: ArrayDict,
                                  mask_token_dict: ArrayDict
                                  ) -> Tuple[ArrayDict, ArrayDict]:
    """"Unshuffle the tokens for modality inpainting.

    Place the masked tokens of one modality (target) and the unmasked tokens
    of another modality in a matrix to create the input for the decoder.
    The unmasked tokens are placed first followed by the masked tokens.

    Args:
      x_unmasked_proj_dict: Dictionary with the unmasked tokens. The shape of
        the unmasked tokens is: [n_batch, n_unmasked_tokens, hidden_size].
      x_tokens_dict: Dictionary with all tokens. Used only for computing the
        total number of tokens. The shape of the tokens is:
        [n_batch, n_tokens, hidden_size].
      mask_token_dict: Dictionary with the masked tokens. The shape of the mask
        token is: [1, 1, hidden_size].

    Returns:
      x_all_dict: Dictionary with the input for the decoder. The shape of the
      tokens is: [n_batch, n_tokens, hidden_size].
    """

    def get_different_key(key_):
      available_keys = set([base_model.FeatureTargets.RGB,
                            base_model.FeatureTargets.SPECTROGRAM])
      return next(iter((available_keys - set([key_]))))

    x_all_dict = {}
    token_mask_dict = {}
    for key in x_unmasked_proj_dict.keys():
      x_unmasked_proj = x_unmasked_proj_dict[get_different_key(key)]
      n_batch = x_unmasked_proj.shape[0]
      batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
      n_tokens = x_tokens_dict[key].shape[1]
      x_all = jnp.zeros((n_batch, n_tokens, self.decoder_config.hidden_size))

      # These are indices that come from the other modality. Note that as shapes
      # between modalities can vary, there is no correspondence between the
      # tokens. Therefore, we place all the unmasked tokens first in the token
      # sequence.
      num_unmasked_tokens = x_unmasked_proj.shape[1]
      unmasked_indices = jnp.repeat(
          jnp.arange(num_unmasked_tokens).reshape(1, num_unmasked_tokens),
          n_batch, axis=0)
      x_all = x_all.at[batch_indices, unmasked_indices].set(x_unmasked_proj)

      # These are indices that will be updated with the current mask token.
      num_masked_tokens = n_tokens - num_unmasked_tokens
      masked_indices = jnp.repeat(
          jnp.arange(num_unmasked_tokens, n_tokens).reshape(
              1, num_masked_tokens), n_batch, axis=0)
      x_all = x_all.at[batch_indices, masked_indices].set(mask_token_dict[key])

      unmasked_token_mask = jnp.zeros((n_batch, num_unmasked_tokens))
      masked_token_mask = jnp.ones((n_batch, num_masked_tokens))
      token_mask = jnp.concatenate((
          unmasked_token_mask, masked_token_mask), axis=1)

      x_all_dict[key] = x_all
      token_mask_dict[key] = token_mask

    return x_all_dict, token_mask_dict

  def unshuffle_tokens(self,
                       x_unmasked_proj_dict: ArrayDict,
                       unmasked_indices_dict: ArrayDict,
                       masked_indices_dict: ArrayDict,
                       x_tokens_dict: ArrayDict,
                       mask_token_dict: ArrayDict
                       ) -> ArrayDict:
    """"Unshuffles the tokens and puts mask tokens at masked indices."""
    # This effectively "unshuffles" the tokens. This means that we can simply
    # add positional encodings in the decoder without having to worry about
    # their ordering.

    x_all_dict = {}
    for key in x_unmasked_proj_dict.keys():
      x_unmasked_proj = x_unmasked_proj_dict[key]
      n_batch = x_unmasked_proj.shape[0]
      batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
      n_tokens = x_tokens_dict[key].shape[1]
      x_all = jnp.zeros((n_batch, n_tokens, self.decoder_config.hidden_size))

      unmasked_indices = unmasked_indices_dict[key]
      masked_indices = masked_indices_dict[key]
      x_all = x_all.at[batch_indices, unmasked_indices].set(x_unmasked_proj)
      x_all = x_all.at[batch_indices, masked_indices].set(mask_token_dict[key])

      x_all_dict[key] = x_all

    return x_all_dict

  def remove_cls_token(self, x_unmasked_proj_dict: ArrayDict
                       ) -> Tuple[ArrayDict, ArrayDict]:
    """"Remove the cls token."""
    cls_encoded_dict = {}
    if self.classifier == 'token':
      raise NotImplementedError('Token classifier is not implemented yet!')
    return cls_encoded_dict, x_unmasked_proj_dict

  def apply_decoder_projection(self,
                               x_unmasked_encoded_dict: ArrayDict
                               ) -> ArrayDict:
    """Project the unmasked tokens to decoder latent space dimension."""
    x_unmasked_proj_dict = {}
    for key, x_unmasked_encoded in x_unmasked_encoded_dict.items():
      x_unmasked_proj = nn.Dense(
          self.decoder_config.hidden_size,
          use_bias=self.decoder_config.get('use_projection_bias', True),
          kernel_init=nn.initializers.xavier_uniform(),
          name=f'decoder_projection_{key}')(x_unmasked_encoded)

      x_unmasked_proj_dict[key] = x_unmasked_proj
    return x_unmasked_proj_dict

  def mask_tokens(self, x_tokens_dict: ArrayDict, train: bool
                  ) -> Tuple[ArrayDict, ArrayDict, ArrayDict, ArrayDict]:
    """Mask the tokens based on their probability."""
    x_unmasked_dict = {}
    token_mask_dict = {}
    unmasked_indices_dict = {}
    masked_indices_dict = {}

    for key, x_tokens in x_tokens_dict.items():
      n_batch, n_tokens = x_tokens.shape[:2]
      if train:
        if self.masking_strategy == 'random':
          # Generate mask indices by randomly masking the tokens.
          n_masked = int(self.token_mask_probability_dict[key] * n_tokens)
          mask_indices, unmasked_indices, token_mask = (
              model_utils.get_mask_indices(
                  n_batch, n_tokens, n_masked, self.make_rng('dropout')
              )
          )
        else:
          raise ValueError(
              f'The masking strategy {self.masking_strategy} is not supported.'
          )
        # Process only unmasked tokens with the encoder.
        batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
        x_unmasked = x_tokens[batch_indices, unmasked_indices]
      else:
        x_unmasked = x_tokens
        token_mask = jnp.zeros((n_batch, n_tokens))
        # We won't need this if train is False.
        unmasked_indices = None
        mask_indices = None

      x_unmasked_dict[key] = x_unmasked
      token_mask_dict[key] = token_mask
      unmasked_indices_dict[key] = unmasked_indices
      masked_indices_dict[key] = mask_indices

    return (x_unmasked_dict, token_mask_dict,  # pytype: disable=bad-return-type  # jax-ndarray
            unmasked_indices_dict, masked_indices_dict)

  def shuffle_tokens_and_token_mask(self,
                                    x_tokens_dict: ArrayDict,
                                    token_mask_dict: ArrayDict,
                                    rng: jax.Array,
                                    ) -> Tuple[ArrayDict, ArrayDict]:
    """Shuffle the tokens and the token masks."""
    # For the inpainting strategy, we add the unmasked token (from a different
    # modality) at the begining of the array and then we complete with unmasked
    # tokens. Now, we shuffle the tokens and the token masks.
    # TODO(lgeorgescu): hard assumption that RGB is in the dict.
    n_batch = x_tokens_dict[base_model.FeatureTargets.RGB].shape[0]
    batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
    rng_keys = jax.random.split(rng, n_batch * len(x_tokens_dict))
    idx_rng_key = 0
    for key in x_tokens_dict:
      x_tokens = x_tokens_dict[key]
      token_mask = token_mask_dict[key]
      n_tokens = x_tokens.shape[1]
      ids = jnp.tile(jnp.arange(n_tokens), n_batch).reshape((n_batch, n_tokens))
      ids = jax.vmap(
          lambda seq, rng: jax.random.permutation(rng, seq, independent=True))(
              ids, rng_keys[idx_rng_key * n_batch: (idx_rng_key + 1) * n_batch])

      x_tokens = x_tokens.at[batch_indices, ids].set(x_tokens)
      token_mask = token_mask.at[batch_indices, ids].set(token_mask)
      x_tokens_dict[key] = x_tokens
      token_mask_dict[key] = token_mask
      idx_rng_key += 1

    return x_tokens_dict, token_mask_dict


def apply_encoder(x_unmasked_dict: ArrayDict,
                  encoder_strategy: str,
                  mlp_dim: int,
                  num_layers: int,
                  num_heads: int,
                  attention_config: ml_collections.ConfigDict,
                  dropout_rate: float,
                  attention_dropout_rate: float,
                  stochastic_droplayer_rate: float,
                  dtype: jnp.dtype,
                  normalise_encoder_output: bool,
                  train: bool,
                  fusion_layers: Optional[int] = None)-> ArrayDict:
  """Apply the encoder for each modality."""

  def separate_encoders():
    x_unmasked_encoded_dict = {}
    for key, x_unmasked_ in x_unmasked_dict.items():
      x_unmasked_encoded = vivit_model.Encoder(
          temporal_dims=None,
          mlp_dim=mlp_dim,
          num_layers=num_layers,
          num_heads=num_heads,
          attention_config=attention_config,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          stochastic_droplayer_rate=stochastic_droplayer_rate,
          dtype=dtype,
          positional_embedding='none',  # Has already been added.
          normalise_output=normalise_encoder_output,
          name=f'Transformer_{key}')(x_unmasked_, train=train)

      x_unmasked_encoded = nn_layers.IdentityLayer(
          name=f'encoder_output_{key}')(x_unmasked_encoded)

      x_unmasked_encoded_dict[key] = x_unmasked_encoded
    return x_unmasked_encoded_dict

  def concat_and_encode():
    x_unmasked_encoded_dict = {}
    encoder = vivit_model.Encoder(
        temporal_dims=None,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        attention_config=attention_config,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        stochastic_droplayer_rate=stochastic_droplayer_rate,
        dtype=dtype,
        positional_embedding='none',  # Has already been added.
        normalise_output=normalise_encoder_output,
        name='Transformer')

    x_unmasked_input_list = []
    x_unmasked_key_list = []
    for key, x_unmasked_ in x_unmasked_dict.items():
      x_unmasked_input_list.append(x_unmasked_)
      x_unmasked_key_list.append(key)

    x_unmasked_concat_input = jnp.concatenate(x_unmasked_input_list, axis=1)
    x_unmasked_encoded_concat = encoder(x_unmasked_concat_input, train=train)
    start_index = 0
    for key, x_unmasked_ in zip(x_unmasked_key_list, x_unmasked_input_list):
      end_index = start_index + x_unmasked_.shape[1]
      x_unmasked_encoded_dict[key] = nn_layers.IdentityLayer(
          name=f'encoder_output_{key}')(
              x_unmasked_encoded_concat[:, start_index:end_index])
      start_index = end_index
      assert x_unmasked_encoded_dict[key].shape[1] == x_unmasked_.shape[1]

    return x_unmasked_encoded_dict

  def same_encoder():
    x_unmasked_encoded_dict = {}
    encoder = vivit_model.Encoder(
        temporal_dims=None,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        attention_config=attention_config,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        stochastic_droplayer_rate=stochastic_droplayer_rate,
        dtype=dtype,
        positional_embedding='none',  # Has already been added.
        normalise_output=normalise_encoder_output,
        name='Transformer')

    for key, x_unmasked_ in x_unmasked_dict.items():
      x_unmasked_encoded = encoder(x_unmasked_, train=train)
      x_unmasked_encoded = nn_layers.IdentityLayer(
          name=f'encoder_output_{key}')(x_unmasked_encoded)
      x_unmasked_encoded_dict[key] = x_unmasked_encoded
    return x_unmasked_encoded_dict

  def separate_encoders_and_concat():
    assert fusion_layers is not None
    num_layers_single = num_layers - fusion_layers
    num_layers_concat = fusion_layers
    assert num_layers_single + num_layers_concat == num_layers

    x_unmasked_encoded_dict = {}

    for key, x_unmasked_ in x_unmasked_dict.items():
      x_unmasked_encoded = vivit_model.Encoder(
          temporal_dims=None,
          mlp_dim=mlp_dim,
          num_layers=num_layers_single,
          num_heads=num_heads,
          attention_config=attention_config,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          stochastic_droplayer_rate=stochastic_droplayer_rate,
          dtype=dtype,
          positional_embedding='none',  # Has already been added.
          normalise_output=normalise_encoder_output,
          name=f'Transformer_{key}')(x_unmasked_, train=train)

      x_unmasked_encoded_dict[key] = x_unmasked_encoded

    encoder_concat = vivit_model.Encoder(
        temporal_dims=None,
        mlp_dim=mlp_dim,
        num_layers=num_layers_concat,
        num_heads=num_heads,
        attention_config=attention_config,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        stochastic_droplayer_rate=stochastic_droplayer_rate,
        dtype=dtype,
        positional_embedding='none',
        normalise_output=normalise_encoder_output,
        name='Transformer_concat')

    x_unmasked_input_list = []
    x_unmasked_key_list = []
    for key, x_unmasked_encoded_ in x_unmasked_encoded_dict.items():
      x_unmasked_input_list.append(x_unmasked_encoded_)
      x_unmasked_key_list.append(key)

    x_unmasked_concat_input = jnp.concatenate(x_unmasked_input_list, axis=1)
    x_unmasked_encoded_concat = encoder_concat(x_unmasked_concat_input,
                                               train=train)

    x_unmasked_encoded_dict_out = {}
    start_index = 0
    for key, x_unmasked_ in zip(x_unmasked_key_list, x_unmasked_input_list):
      end_index = start_index + x_unmasked_.shape[1]
      x_unmasked_encoded_dict_out[key] = nn_layers.IdentityLayer(
          name=f'encoder_output_{key}')(
              x_unmasked_encoded_concat[:, start_index:end_index])
      start_index = end_index
      assert x_unmasked_encoded_dict_out[key].shape[1] == x_unmasked_.shape[1]

    return x_unmasked_encoded_dict_out

  if encoder_strategy == 'separate_encoders':
    return separate_encoders()
  elif encoder_strategy == 'concat_and_encode':
    return concat_and_encode()
  elif encoder_strategy == 'same_encoder':
    return same_encoder()
  elif encoder_strategy == 'separate_encoders_and_concat':
    return separate_encoders_and_concat()
  else:
    raise ValueError(
        f'The encoder strategy {encoder_strategy} is not supported!')


def add_cls_token(x_all_dict: ArrayDict, classifier: str,
                  cls_encoded_dict: Optional[ArrayDict] = None
                  ) -> ArrayDict:
  """Add the cls token."""
  # If cls_encoded_dict is None, then the cls token will be generated.
  # Otherwise with be added back to the matrix.
  if cls_encoded_dict is None:
    # generate the cls token
    pass
  if classifier == 'token':
    raise NotImplementedError('Token classifer is not implemented yet!')

  return x_all_dict


def add_positional_embeddings(x_tokens_dict, positional_embedding):
  """Add positional encodings."""
  if positional_embedding in ['sinusoidal_1d', 'learned_1d']:
    for key, x_tokens_ in x_tokens_dict.items():
      x_tokens_ = model_utils.add_positional_embeddings(
          x_tokens_, positional_embedding, input_shape=x_tokens_.shape,
          layer_name=f'posembed_input_{key}')
      x_tokens_dict[key] = x_tokens_
  else:
    raise ValueError('Only 1d positional embdedding are supported!')

  return x_tokens_dict


def tokenize_input(inputs: ArrayDict, temporal_encoding_config: str,
                   patches: ml_collections.ConfigDict, hidden_size: int,
                  ) -> ArrayDict:
  """Tokenize the input based on their modality."""
  embed_2d = {'spectrogram'}
  temporal_encode = {'RGB', 'modis', 'l7', 's2', 's1', 'nicfi', 'nicfi_monthly',
                     'alos'}
  if 'size' in patches:  # Handles case of one patch given for all modalities.
    modal_patches = {key: patches for key in inputs}
  else:
    modal_patches = patches
  tokens_dict = {}
  for key in inputs:
    # Shape is [batch, num_tokens, hidden_size].
    if key in temporal_encode:
      x_tokens, _ = vivit_model.temporal_encode(
          inputs[key],
          temporal_encoding_config,
          patches=modal_patches[key],
          hidden_size=hidden_size,
          name=f'embedding_{key}',
      )
    elif key in embed_2d:
      x_tokens = model_utils.embed_2d_patch(
          inputs[key],
          patches=modal_patches[key],
          embedding_dim=hidden_size,
          name=f'embedding_{key}',
      )
    else:
      raise ValueError(f'Modality {key} is not supported!')
    tokens_dict[key] = x_tokens

  return tokens_dict


class ViViTMultiMaskedAutoencoderModel(base_model.MaskedFeatureRegressionModel):
  """ViViT model for multi-modalitymasked autoencoder pretraining."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))

    num_classes_dict = {}
    for feature_target in self.config.masked_feature_loss.target:
      if feature_target == base_model.FeatureTargets.RGB:
        select_central_frame = (
            self.config.masked_feature_loss.select_central_frame)
        patch_size = tuple(self.config.model.patches.size)
        channels = 3
      elif feature_target == base_model.FeatureTargets.SPECTROGRAM:
        patch_size = tuple(self.config.model.patches.size[:2])
        select_central_frame = False
        channels = 1
      else:
        raise ValueError(f'{feature_target} is not supported!')

      num_classes = base_model.get_output_shapes(feature_target,
                                                 patch_size,
                                                 select_central_frame,
                                                 channels=channels)
      num_classes_dict[feature_target] = num_classes

    return ViViTMultiMaskedAutoencoder(
        num_classes_dict=num_classes_dict,
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        token_mask_probability_dict=(
            self.config.masked_feature_loss.token_mask_probability_dict),
        masking_strategy=self.config.masked_feature_loss.get('masking_strategy',
                                                             'random'),
        temporal_encoding_config=self.config.model.temporal_encoding_config,
        attention_config=self.config.model.attention_config,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.model.get(
            'attention_dropout_rate', 0.1),
        stochastic_droplayer_rate=self.config.model.get(
            'stochastic_droplayer_rate', 0),
        dtype=model_dtype,
        decoder_config=self.config.model.get('decoder_config', None),
        positional_embedding=self.config.model.get('positional_embedding',
                                                   'sinusoidal_1d'),
        positional_embedding_decoder=self.config.model
        .get('positional_embedding_decoder', 'sinusoidal_1d'),
        normalise_encoder_output=self.config.model.get(
            'normalise_encoder_output', True),
        use_inpainting=self.config.model.get('use_inpainting', False),
        shuffle_inpainted_tokens=self.config.model.get(
            'shuffle_inpainted_tokens', False),
        encoder_strategy=self.config.model.get('encoder_strategy'),
        decoder_strategy=self.config.model.get('decoder_strategy'),
        use_modality_tokens=self.config.model.get('use_modality_tokens', False),
        fusion_layers=self.config.model.get('fusion_layers', None)
    )

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    return vivit_model_utils.initialise_from_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj)

  def loss_function(self,  # pytype: disable=signature-mismatch  # jax-ndarray
                    predictions: ArrayDict,
                    prediction_masks: ArrayDict,
                    batch: base_model_lib.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the (weighted) mean squared error.

    Args:
      predictions: Dictionary with the output of model in shape
        [batch, num_tokens, channels].
      prediction_masks: Dictionary with the tokens to compute the loss on.
        Shape is [batch, num_tokens]
      batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        L2 regularization.

    Returns:
      The (weighted) mean squared error.
    """

    def get_loss_weights(target_name_):
      batch_mask = batch.get('batch_mask')
      if batch_mask is None:
        batch_mask = jnp.ones(prediction_masks[target_name_].shape)
      if batch_mask.ndim == 1:
        batch_mask = jnp.expand_dims(batch_mask, axis=-1)
      if self.config.masked_feature_loss.get('loss_unmasked_tokens', False):
        loss_weights = batch_mask
      else:
        loss_weights = batch_mask * prediction_masks[target_name_]

      return loss_weights

    total_loss = 0.0
    for target_name in self.config.masked_feature_loss.target:
      targets = batch['targets'][target_name]
      loss_weights = get_loss_weights(target_name)
      loss = model_utils_lib.weighted_mean_squared_error(
          predictions[target_name], targets, loss_weights, axis=-1)

      # Mean squared error is normalised by the number of tokens.
      # If this option is enabled, we normalise further by the number
      # of features we are regressing to.
      if self.config.masked_feature_loss.get('normalise_by_output_dimension',
                                             False):
        output_dimension = predictions[target_name].shape[-1]
        loss = loss / output_dimension
      total_loss = total_loss + (
          loss * self.config.masked_feature_loss.modality_weight[target_name])

    if self.config.get('l2_decay_factor'):
      l2_loss = model_utils_lib.l2_regularization(model_params)
      total_loss += 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None
                     ) -> base_model_lib.MetricFn:
    """Returns a callable metric function for the model.

    By default, we return the same metric for each split.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API:
    ```metrics_fn(predictions, batch)```
    """

    del split  # Same function for all splits.
    return functools.partial(
        feature_regression_metrics_function,
        feature_target=self.config.masked_feature_loss.target,
        metrics=base_model._REGRESSION_METRICS)  # pylint: disable=protected-access


def feature_regression_metrics_function(
    predictions: ArrayDict, prediction_masks: ArrayDict,
    batch: base_model_lib.Batch, feature_target: str,
    metrics: base_model_lib.MetricNormalizerFnDict,
    ) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for the feature regression task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(predictions, targets, weights)```
  and returns an array of shape [batch,]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
    predictions:  Dictionary with the output of model in shape [batch, length].
    prediction_masks:  Dictionary of the predictions which are valid.
    batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
    feature_target: The feature targets used for feature regression.
    metrics: The regression metrics to evaluate. The key is the
      name of the  metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  evaluated_metrics = {}
  # TODO(lgeorgescu): add the weighted sum in the evaluation too

  for key_target in feature_target:
    batch_mask = batch.get('batch_mask')
    if batch_mask is None:
      batch_mask = jnp.ones(prediction_masks[key_target].shape)
    if batch_mask.ndim == 1:
      n_batch = predictions[key_target].shape[0]
      batch_mask = jnp.reshape(batch_mask, (n_batch, 1))
    weights = batch_mask * prediction_masks[key_target]

    for key, val in metrics.items():
      evaluated_metrics[key + '_' + key_target] = (
          model_utils_lib.psum_metric_normalizer(
              (val[0](predictions[key_target], batch['targets'][key_target],  # pytype: disable=wrong-arg-types  # jax-ndarray
                      weights),
               val[1](predictions[key_target], batch['targets'][key_target],  # pytype: disable=wrong-arg-types  # jax-ndarray
                      weights))))

  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray
