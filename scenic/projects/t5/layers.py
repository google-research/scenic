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

"""A wrapper class for T5 model.
"""
import functools
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.model_lib.base_models import base_model
from t5x import decoding
from t5x.examples.t5 import layers as t5_layers
from t5x.examples.t5 import network as t5
from t5x.models import DecodeFnCallable

beam_search = decoding.beam_search
temperature_sample = decoding.temperature_sample

Batch = Dict[str, jnp.ndarray]
PyTree = Any


class T5(nn.Module):
  """T5 model consisting of encoder and decoder transformers.

  This class simply wraps network.Transformer class in t5x.examples.t5.

  Attributes:
    vocab_size: Size of the vocabulary.
    emb_dim: Size of the embeddings.
    num_heads: Number of attention heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    head_dim: Size of the embeddings in each head.
    mlp_dim: Size of the MLP output embeddings.
    dropout_rate: Dropout rate.
    dtype: Data type.
    mlp_activations: Sequence of activations in MLP.
    logits_via_embedding: Use the embedding weights for computing logits.
  """
  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False
  float32_attention_logits: bool = False

  def setup(self):
    self.t5_config = t5.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
        float32_attention_logits=self.float32_attention_logits)
    self.t5_module = t5.Transformer(self.t5_config)

  def encode(self,
             encoder_input_tokens: jnp.ndarray,
             encoder_segment_ids: Optional[jnp.ndarray] = None,
             enable_dropout: bool = True):
    return self.t5_module.encode(encoder_input_tokens,
                                 encoder_segment_ids,
                                 enable_dropout)

  def decode(
      self,
      encoded: jnp.ndarray,
      encoder_input_tokens: jnp.ndarray,  # Only needed for masks.
      decoder_input_tokens: jnp.ndarray,
      decoder_target_tokens: jnp.ndarray,
      encoder_segment_ids: Optional[jnp.ndarray] = None,
      decoder_segment_ids: Optional[jnp.ndarray] = None,
      decoder_positions: Optional[jnp.ndarray] = None,
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None):
    return self.t5_module.decode(encoded,
                                 encoder_input_tokens,
                                 decoder_input_tokens,
                                 decoder_target_tokens,
                                 encoder_segment_ids,
                                 decoder_segment_ids,
                                 decoder_positions,
                                 enable_dropout,
                                 decode,
                                 max_decode_length)

  def __call__(self,
               encoder_input_tokens: jnp.ndarray,
               decoder_input_tokens: jnp.ndarray,
               decoder_target_tokens: jnp.ndarray,
               encoder_segment_ids: Optional[jnp.ndarray] = None,
               decoder_segment_ids: Optional[jnp.ndarray] = None,
               encoder_positions: Optional[jnp.ndarray] = None,
               decoder_positions: Optional[jnp.ndarray] = None,
               *,
               enable_dropout: bool = True,
               decode: bool = False):
    """Applies T5 model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      logits array from full transformer.
    """
    return self.t5_module(encoder_input_tokens,
                          decoder_input_tokens,
                          decoder_target_tokens,
                          encoder_segment_ids,
                          decoder_segment_ids,
                          encoder_positions,
                          decoder_positions,
                          enable_dropout=enable_dropout,
                          decode=decode)


class T5Encoder(nn.Module):
  """T5 encoder as a separate model.

  This module contains the encoder part of a pretrained T5. It is useful when
  adopting the pretrained T5 encoder as a part of a larger network. Note that
  the embedding layer should be created outside the module and provided as a
  parameter `shared_embedding` to share it in other parts of the network (e.g.,
  text encoder). If `shared_embedding` is not provided, the embedding layer is
  created within the module.

  Attributes:
    vocab_size: Size of the vocabulary.
    emb_dim: Size of the embeddings.
    num_heads: Number of attention heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    head_dim: Size of the embeddings in each head.
    mlp_dim: Size of the MLP output embeddings.
    dropout_rate: Dropout rate.
    dtype: Data type.
    mlp_activations: Sequence of activations in MLP.
    logits_via_embedding: Use the embedding weights for computing logits.
    shared_embedding: Optional. Embedding layer that is shared outside this
      module. If not given, a non-shared embedding layer will be created within
      the module.
  """
  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False
  shared_embedding: Optional[nn.Module] = None
  float32_attention_logits: bool = False

  def setup(self):
    self.t5_config = t5.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
        float32_attention_logits=self.float32_attention_logits)
    if self.shared_embedding is None:
      self.nonshared_embedding = t5_layers.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          dtype=self.dtype,
          attend_dtype=jnp.float32,  # For logit training stability.
          embedding_init=nn.initializers.normal(stddev=1.0),
          one_hot=True,
          name='token_embedder')
      embedding_layer = self.nonshared_embedding
    else:
      embedding_layer = self.shared_embedding
    self.encoder_module = t5.Encoder(self.t5_config, embedding_layer)

  def __call__(self,
               encoder_input_tokens,
               encoder_segment_ids=None,
               enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.t5_config
    assert encoder_input_tokens.ndim == 2  # (batch, len)

    # Make padding attention mask.
    encoder_mask = t5_layers.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      encoder_mask = t5_layers.combine_masks(
          encoder_mask,
          t5_layers.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    return self.encoder_module(
        encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)


class T5Decoder(nn.Module):
  """T5 decoder as a separate model.

  This module contains the decoder part of a pretrained T5. It is useful when
  adopting the pretrained T5 decoder as a part of a larger network. Note that
  the embedding layer should be created outside the module and provided as a
  parameter `shared_embedding` to share it in other parts of the network (e.g.,
  text encoder). If `shared_embedding` is not provided, the embedding layer is
  created within the module.

  Attributes:
    vocab_size: Size of the vocabulary.
    emb_dim: Size of the embeddings.
    num_heads: Number of attention heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    head_dim: Size of the embeddings in each head.
    mlp_dim: Size of the MLP output embeddings.
    dropout_rate: Dropout rate.
    dtype: Data type.
    mlp_activations: Sequence of activations in MLP.
    logits_via_embedding: Use the embedding weights for computing logits.
    shared_embedding: Optional. Embedding layer that is shared outside this
      module. If not given, a non-shared embedding layer will be created within
      the module.
  """
  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False
  shared_embedding: Optional[nn.Module] = None
  float32_attention_logits: bool = False

  def setup(self):
    self.t5_config = t5.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
        float32_attention_logits=self.float32_attention_logits)
    if self.shared_embedding is None:
      self.nonshared_embedding = t5_layers.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          dtype=self.dtype,
          attend_dtype=jnp.float32,  # For logit training stability.
          embedding_init=nn.initializers.normal(stddev=1.0),
          one_hot=True,
          name='token_embedder')
      embedding_layer = self.nonshared_embedding
    else:
      embedding_layer = self.shared_embedding
    self.decoder_module = t5.Decoder(self.t5_config, embedding_layer)

  def __call__(self,
               encoded,
               encoder_input_tokens: jnp.ndarray,  # Only needed for masks.
               decoder_input_tokens: jnp.ndarray,
               decoder_target_tokens: jnp.ndarray,
               encoder_segment_ids: Optional[jnp.ndarray] = None,
               decoder_segment_ids: Optional[jnp.ndarray] = None,
               encoder_positions: Optional[jnp.ndarray] = None,
               decoder_positions: Optional[jnp.ndarray] = None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None):
    """Decode from the given encoded embedding using a T5 decoder.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.
    This function is a copy of the decode() method of t5.Transformer.

    Args:
      encoded: input embeddings obtained from an encoder.
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: Maximum length for autoregressive decoding.

    Returns:
      logits array from full transformer.
    """

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = t5_layers.make_attention_mask(
          jnp.ones_like(decoder_target_tokens),
          encoder_input_tokens > 0,
          dtype=self.dtype)
    else:
      decoder_mask = t5_layers.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=self.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = t5_layers.make_attention_mask(
          decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=self.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

      encoder_decoder_mask = t5_layers.combine_masks(
          encoder_decoder_mask,
          t5_layers.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=self.dtype))

    logits = self.decoder_module(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits.astype(self.dtype)


class T5Model(base_model.BaseModel):
  """T5 model implementing autoregressive decoding."""

  def _compute_logits_from_slice(
      self, decoding_state: decoding.DecodingState,
      all_variables: PyTree, encoded_inputs: jnp.ndarray,
      input_masks: jnp.ndarray,
      max_decode_length: int) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache
    flat_logits, new_vars = self.flax_model.apply(
        {
            'cache': flat_cache,
            **all_variables
        },
        encoded_inputs, **{
            'encoder_input_tokens': input_masks,
            'decoder_input_tokens': flat_ids,
            'decoder_target_tokens': flat_ids,
        },
        enable_dropout=False,
        decode=True,
        max_decode_length=max_decode_length,
        mutable=['cache'],
        method=self.flax_model.decode)
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: PyTree,
      decode_fn: DecodeFnCallable,
      eos_id: int = 1,
      decoder_params: Optional[Dict[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
  ):
    """Predict with fast decoding beam search on a batch.

    This is copied and modified from T5X EncoderDecoderTransformer model in
    third_party/py/t5x/models.py.

    Here we refer to "parameters" for values that can be compiled into the
    model dynamically, as opposed to static configuration settings that require
    a recompile. For example, the model weights and the decoder brevity-penalty
    are parameters and can be modified without requiring a recompile. The number
    of layers, the batch size and the decoder beam size are configuration
    options that require recompilation if changed.

    This method can be used with a customizable decoding function as long as it
    follows the signature of `DecodeFnCallable`. In order to provide a unified
    interface for the decoding functions, we use a generic names. For example a
    beam size is a concept unique to beam search. Conceptually, it corresponds
    to the number of sequences returned by the beam search.  Therefore, the
    generic argument `num_decodes` corresponds to the beam size if
    `decode_fn` is a beam search. For temperature sampling, `num_decodes`
    corresponds to the number of indepedent sequences to be sampled. Typically
    `num_decodes = 1` is used for tempeature sampling.

    If `return_all_decodes = True`, the return tuple contains the predictions
    with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
    probability of the generated sequence) with a shape [batch, num_decodes].

    If `return_all_decodes = False`, the return tuple contains the predictions
    with a shape [batch, max_decode_len] and the scores with a shape [batch].

    `decoder_params` can be used to pass dynamic configurations to
    `decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    Args:
      params: model parameters.
      batch: a batch of inputs. It's a nested Pytree with two keys:
        `encoder_inputs` and `decoder_inputs`, each of which contains the
        default T5 encoder and decoder params.
      decode_fn: function implementing the decode method.
      eos_id: EOS token id in the vocabulary.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    # Prepare zeroed-out autoregressive cache.
    encoder_inputs = jax.tree_util.tree_map(jnp.ones_like,
                                            batch['encoder_inputs'])
    decoder_inputs = jax.tree_util.tree_map(jnp.ones_like,
                                            batch['decoder_inputs'])
    _, variables_with_cache = self.flax_model.apply(
        params,
        # encoder_input_tokens=encoder_inputs,
        # decoder_input_tokens=decoder_inputs,
        # decoder_target_tokens=decoder_inputs,
        **encoder_inputs,
        **decoder_inputs,
        decode=True,
        enable_dropout=False,
        mutable=['cache'])
    cache = variables_with_cache['cache']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    beam_expand_fn = functools.partial(
        decoding.flat_batch_beam_expand, beam_size=num_decodes)
    non_expanded_encoded = self.flax_model.apply(
        params,
        **batch['encoder_inputs'],
        enable_dropout=False,
        method=self.flax_model.encode)
    encoded_inputs = jax.tree_util.tree_map(beam_expand_fn,
                                            non_expanded_encoded)

    # Set the all output embeddings to be valid inputs if encoder_input_tokens
    # are not provided. Note that this tensor should be beam-extended too.
    if 'encoder_input_tokens' not in decoder_inputs:
      input_masks = jnp.ones(encoded_inputs.shape[:-1])
    else:
      input_masks = jax.tree_util.tree_map(
          beam_expand_fn, decoder_inputs['encoder_input_tokens'])

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        all_variables=params,
        encoded_inputs=encoded_inputs,
        input_masks=input_masks,
        max_decode_length=decoder_inputs['decoder_input_tokens'].shape[1])

    if decoder_params is None:
      decoder_params = {}

    # For beam search, `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information. For temperature sampling,
    # `decod_prompt_inputs` will be filled with the sampled ids.
    decoder_prompt_inputs = jnp.zeros_like(
        decoder_inputs['decoder_input_tokens'])

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    decodes, scores = decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=eos_id,
        num_decodes=num_decodes,
        cache_offset=0,
        **decoder_params)

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1]}

  def build_flax_model(self) -> nn.Module:
    return T5(**self.config)
