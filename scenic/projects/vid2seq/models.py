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

"""Dual encoder w/ temporal transformer + T5 encoder and decoder w/ time tokens.
"""

import functools
from typing import Any, Dict, Mapping, Optional, Tuple

from absl import logging
import flax.linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils as base_model_utils
from scenic.projects.baselines import vit
from scenic.projects.t5 import layers as t5_model
from scenic.projects.t5 import model as t5_pretrained
from t5x import decoding
from t5x.models import DecodeFnCallable


beam_search = decoding.beam_search
temperature_sample = decoding.temperature_sample

Batch = Dict[str, jnp.ndarray]
PyTree = Any


class CatEncoder(nn.Module):
  """Concat ViT temporal encodings with T5 text encodings."""
  enc_type: str
  enc_config: ml_collections.ConfigDict
  embedder: nn.Module
  num_bins: int

  def setup(self):
    self.visual_encoder = vit.Encoder(
        mlp_dim=self.enc_config.get('dim', 2048),
        num_layers=self.enc_config.get('layers', 12),
        num_heads=self.enc_config.get('heads', 12),
        positional_embedding=self.enc_config.get('pos_embed', 'learned_1d'),
        dropout_rate=self.enc_config.get('dropout_rate', 0.),
        attention_dropout_rate=self.enc_config.get('dropout_rate', 0.),
        stochastic_depth=self.enc_config.get('stochastic_depth', 0.))
    enc_cfg = self.enc_config.get('pretrained_config', 't5_1_1_base')
    t5_config = t5_pretrained.CONFIGS[enc_cfg]
    t5_config['dropout_rate'] = self.enc_config.get('t5_dropout_rate', 0.)
    if self.num_bins:  # add time tokens to the vocabulary
      t5_config['vocab_size'] = 32128 + self.num_bins
    self.t5_encoder = t5_model.T5Encoder(
        **t5_config,
        shared_embedding=self.embedder,
        name='video_encoder')
    self.proj_dim = 768
    if enc_cfg == 't5_1_1_large':
      self.proj_dim = 1024
      self.proj = nn.Dense(
          self.proj_dim, dtype=self.t5_encoder.dtype, name='vis_to_text')
    elif enc_cfg == 't5_1_1_small':
      self.proj_dim = 512
      self.proj = nn.Dense(
          self.proj_dim, dtype=self.t5_encoder.dtype, name='vis_to_text')

  def __call__(self,
               features=None,
               encoder_input_tokens=None,
               encoder_segment_ids=None,
               enable_dropout=True):
    if features is not None:
      visual_embeddings = self.visual_encoder(
          features, train=enable_dropout)
      if self.proj_dim != 768:
        visual_embeddings = self.proj(visual_embeddings)
      if encoder_input_tokens is not None:
        x = self.t5_encoder(
            encoder_input_tokens=encoder_input_tokens,
            encoder_segment_ids=None,
            enable_dropout=enable_dropout)
        x = {'encoded': x, 'mask': encoder_input_tokens > 0}
        cat = jnp.concatenate([visual_embeddings, x['encoded']], axis=1)
        cat_mask = jnp.concatenate([
            jnp.ones(visual_embeddings.shape[:2]) > 0,
            x['mask']
        ],
                                   axis=1)
      else:
        cat = visual_embeddings
        cat_mask = jnp.ones(
            visual_embeddings.shape[:2]) > 0
    elif encoder_input_tokens is not None:
      x = self.t5_encoder(
          encoder_input_tokens=encoder_input_tokens,
          encoder_segment_ids=None,
          enable_dropout=enable_dropout)
      cat = x
      cat_mask = encoder_input_tokens > 0
    else:
      raise NotImplementedError
    return {'encoded': cat, 'mask': cat_mask}


class EncoderDecoderModule(nn.Module):
  """Encoder-Decoder module."""

  config: ml_collections.ConfigDict

  def encode(self, *args, **kwargs):
    raise NotImplementedError('Subclasses must implement encode.')

  def decode(self, *args, **kwargs):
    raise NotImplementedError('Subclasses must implement decode.')


class DenseVideoCaptioningModule(EncoderDecoderModule):
  """Dense video captioning module that encodes a video and generate tokens."""

  def _get_encoder(self,
                   enc_type: str,
                   enc_config: ml_collections.ConfigDict,
                   embedder: Optional[nn.Module] = None,
                   num_bins: int = 0):
    if enc_type == 'tmp':
      encoder = vit.Encoder(
          mlp_dim=enc_config.get('dim'),
          num_layers=enc_config.get('layers'),
          num_heads=enc_config.get('heads'),
          positional_embedding=enc_config.get('pos_embed'),
          dropout_rate=enc_config.get('dropout_rate'),
          attention_dropout_rate=enc_config.get('dropout_rate'),
          stochastic_depth=enc_config.get('stochastic_depth'))
    elif enc_type == 't5_encoder':
      t5_config = t5_pretrained.CONFIGS[enc_config.pretrained_config]
      t5_config['dropout_rate'] = enc_config.get('dropout_rate')
      if num_bins:  # add timestamp tokens to the vocabulary
        t5_config['vocab_size'] = 32128 + num_bins
      encoder = t5_model.T5Encoder(
          **t5_config,
          shared_embedding=embedder,
          name='video_encoder')
    elif enc_type == 'cat_encoder':
      encoder = CatEncoder(enc_type=enc_type,
                           enc_config=enc_config,
                           embedder=embedder,
                           num_bins=num_bins)
    else:
      raise ValueError(f'Unrecognized encoder type: {enc_type}.')

    return encoder

  def _get_decoder(self,
                   dec_type: str,
                   dec_config: ml_collections.ConfigDict,
                   num_bins: int,
                   tmp_only: bool = False):
    if dec_type == 't5_decoder':  # add timestamp tokens to the vocabulary
      t5_config = t5_pretrained.CONFIGS[dec_config.pretrained_config]
      t5_config['dropout_rate'] = dec_config.dropout_rate
      t5_config['logits_via_embedding'] = dec_config.logits_via_embedding
      if tmp_only:
        t5_config['vocab_size'] = num_bins + 2
      else:
        t5_config['vocab_size'] = 32128 + num_bins
      decoder_embedder = t5_model.t5_layers.Embed(
          num_embeddings=t5_config['vocab_size'],
          features=t5_config['emb_dim'],
          dtype=t5_config['dtype'],
          attend_dtype=jnp.float32,  # For logit training stability.
          embedding_init=nn.initializers.normal(stddev=1.0),
          one_hot=True,
          name='shared_decoder_token_embedder')
      decoder = t5_model.T5Decoder(
          **t5_config,
          shared_embedding=decoder_embedder,
          name='text_decoder')
    else:
      raise ValueError(f'Unrecognized decoder type: {dec_type}.')

    return (decoder_embedder, decoder)

  def setup(self):
    self.decoder_type = self.config.get('decoder_type', 't5_decoder')
    decoder_config = self.config.decoder.get(self.decoder_type)
    num_bins = self.config.decoder.get('num_bins')
    self.encoder_type = self.config.encoder.get('encoder_type')
    encoder_config = self.config.encoder.get(self.encoder_type)
    self.embedder, self.decoder = self._get_decoder(
        self.decoder_type,
        decoder_config,
        num_bins)

    self.encoder = self._get_encoder(
        self.encoder_type,
        encoder_config,
        self.embedder,
        num_bins)

  def encode(self, encoder_inputs, *, train=True):
    # load modalities
    if 'features' in encoder_inputs:
      features = encoder_inputs['features']
    else:
      features = None

    if 'text' in encoder_inputs:
      encoder_input_tokens = encoder_inputs['text']
    else:
      encoder_input_tokens = None

    if self.config.encoder.encoder_type in [
        't5_encoder', 'cat_encoder'
    ]:  # give correct arguments
      return self.encoder(
          features=features,
          encoder_input_tokens=encoder_input_tokens,
          encoder_segment_ids=None,
          enable_dropout=train)  # pytype: disable=wrong-keyword-args
    return self.encoder(features, train=train)  # pytype: disable=wrong-keyword-args

  def decode(self,
             encoded,
             decoder_inputs,
             *,
             train=True,
             decode=False,
             max_decode_length=None):

    return self.decoder(
        encoded,
        **decoder_inputs,
        enable_dropout=train,
        decode=decode,
        max_decode_length=max_decode_length)

  def __call__(self,
               encoder_inputs,
               decoder_inputs,
               *,
               train=True,
               decode=False,
               max_decode_length=None,
               debug: bool = False):
    if debug:
      logging.info('encoder_inputs: %s', encoder_inputs)
      logging.info('decoder_inputs: %s', decoder_inputs)
    encoded = self.encode(encoder_inputs, train=train)
    # Fill in encoder_input_tokens if not provided.
    # This sets the all output embeddings to be valid inputs.
    if self.config.encoder.encoder_type in [
        't5_encoder', 'cat_encoder'
    ]:
      decoder_inputs['encoder_input_tokens'] = encoded['mask']
      encoded = encoded['encoded']
    else:
      if 'encoder_input_tokens' not in decoder_inputs:
        decoder_inputs['encoder_input_tokens'] = jnp.ones(encoded.shape[:-1])  # pytype: disable=attribute-error

    # joint time and text
    return {'logits': self.decode(
        encoded,
        decoder_inputs,
        train=train,
        decode=decode,
        max_decode_length=max_decode_length),
            'encoded': encoded}


class EncoderWithT5DecoderModel(base_model.BaseModel):
  """Encoder-decoder model with T5 decoder implementing beam-search."""

  def _compute_logits_from_slice(
      self, decoding_state: decoding.DecodingState, all_variables: PyTree,
      encoded_inputs: jnp.ndarray, input_masks: jnp.ndarray,
      max_decode_length: int,
      ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache
    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]
    flat_logits, new_vars = self.flax_model.apply(
        {
            'cache': flat_cache,
            **all_variables
        },
        encoded_inputs, {
            'encoder_input_tokens': input_masks,
            'decoder_input_tokens': flat_ids,
            'decoder_target_tokens': flat_ids,
        },
        train=False,
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
      alpha: float = 0.6,
      decoding_method: str = 'beamsearch',
      temperature: float = 1.0,
      vocabulary_size: int = 32128,
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
      batch: a batch of inputs.
      decode_fn: function implementing the decode method.
      eos_id: EOS token id in the vocabulary.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      alpha: length penalty factor for beam search.
      decoding_method: decoding method.
      temperature: temperature for nucleus sampling.
      vocabulary_size: size of the vocabulary for the textual tokens.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """
    # Prepare zeroed-out autoregressive cache.
    encoder_inputs = jax.tree_util.tree_map(
        jnp.ones_like, batch['encoder_inputs']
    )
    decoder_inputs = jax.tree_util.tree_map(
        jnp.ones_like, batch['decoder_inputs']
    )
    _, variables_with_cache = self.flax_model.apply(
        params,
        encoder_inputs,
        decoder_inputs,
        decode=True,
        train=False,
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
        batch['encoder_inputs'],
        train=False,
        method=self.flax_model.encode)
    encoded_inputs = jax.tree_util.tree_map(
        beam_expand_fn, non_expanded_encoded
    )
    if isinstance(encoded_inputs, dict):  # set decoder mask
      batch['decoder_inputs']['encoder_input_tokens'] = encoded_inputs['mask']
      encoded_inputs = encoded_inputs['encoded']

    # Set the all output embeddings to be valid inputs if encoder_input_tokens
    # are not provided. Note that this tensor should be beam-extended too.
    decoder_inputs = batch['decoder_inputs']
    if 'encoder_input_tokens' not in decoder_inputs:
      input_masks = jnp.ones(encoded_inputs.shape[:-1])
    else:
      input_masks = decoder_inputs['encoder_input_tokens']

    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        all_variables=params,
        encoded_inputs=encoded_inputs,
        input_masks=input_masks,
        max_decode_length=decoder_inputs['decoder_input_tokens'].shape[1])

    if decoder_params is None:
      decoder_params = {}

    # `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information here.
    decoder_prompt_inputs = jnp.zeros_like(
        decoder_inputs['decoder_input_tokens'])

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    if decoding_method == 'temperature_sample':
      decodes, scores = decode_fn(
          inputs=decoder_prompt_inputs,
          cache=cache,
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=eos_id,
          topp=alpha,
          topk=0,
          temperature=temperature,
          num_decodes=num_decodes,
          cache_offset=0,
          **decoder_params)
    else:  # beam search
      decodes, scores = decode_fn(
          inputs=decoder_prompt_inputs,
          cache=cache,
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=eos_id,
          alpha=alpha,
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


def l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.
  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.
  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class DenseVideoCaptioningModel(EncoderWithT5DecoderModel):
  """Dense video captioning model with a video encoder and a text decoder."""

  def build_flax_model(self) -> nn.Module:
    return DenseVideoCaptioningModule(self.config.model)

  def loss_function(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      logits: jnp.ndarray,
      batch: Batch,
      model_params: Optional[Dict[str, jnp.ndarray]] = None) -> float:
    """Returns negative loglikelihood (NLL) of the target sentence with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_voca].
      batch: Batch of data that has 'decoder_target_tokens'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    targets = batch['decoder_inputs']['decoder_target_tokens']

    if logits.ndim != targets.ndim + 1:
      raise ValueError(
          'Incorrect shapes. Got shape %s logits and %s targets' %
          (str(logits.shape), str(targets.shape)))

    target_masks = targets > 0
    vocab_size = logits.shape[-1]
    onehot_targets = common_utils.onehot(targets, vocab_size)

    sent_nll_loss = base_model_utils.weighted_softmax_cross_entropy(
        logits,
        onehot_targets,
        target_masks,
        label_smoothing=self.config.get('label_smoothing'))

    if self.config.get('l2_decay_factor') is None:
      total_loss = sent_nll_loss
    else:
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss = sent_nll_loss + self.config.l2_decay_factor * l2_loss

    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(preds,
      label, weights)```
    """

    def token_accuracy(logits,
                       batch: Batch) -> Dict[str, Tuple[float, int]]:
      # This metric is computed only during teacher forcing training mode.
      if split != 'train':
        return {}

      targets = batch['decoder_inputs']['decoder_target_tokens']

      batch_mask = batch['batch_mask']

      # logits: [batch_size, seq_len, vocab_size]
      # targets: [batch_size, seq_len]
      # batch_mask: [batch_size]
      one_hot_targets = common_utils.onehot(targets, logits.shape[-1])
      masks = jnp.greater(targets, 0).astype(jnp.int32) * batch_mask[:, None]

      n_corrects = base_model_utils.weighted_correctly_classified(
          logits, one_hot_targets, masks)
      n_valids = base_model_utils.num_examples(logits, one_hot_targets, masks)

      key = 'token_accuracy'
      return {  # pytype: disable=bad-return-type  # jax-ndarray
          key:
              base_model_utils.psum_metric_normalizer((n_corrects, n_valids))
      }

    return token_accuracy

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({})
