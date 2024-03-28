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

"""Transformer-based sequence-to-sequence model for video inputs.

Based on third_party/py/flax/examples/wmt/models.py
"""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from typing import Any, Dict, Optional, Tuple

from absl import logging
from flax import linen as nn
from flax.training import common_utils
from immutabledict import immutabledict
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.mbt import model as mbt_model
from scenic.projects.mbt.model import temporal_encode

# Standard default metrics for the classification models.
_CLASSIFICATION_METRICS = immutabledict({
    'accuracy':
        (model_utils.weighted_correctly_classified, model_utils.num_examples),
    'loss': (model_utils.weighted_unnormalized_softmax_cross_entropy,
             model_utils.num_examples)
})


def shift_right(x, axis=1):
  """Shift the input to the right for a given axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  slicing = [slice(None)] * len(x.shape)
  slicing[axis] = slice(0, -1)
  return padded[tuple(slicing)]


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2:2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: hyperparameters of the module
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, inputs, inputs_positions=None, decode=False):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      decode: whether to run in single-position autoregressive mode.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.get('posemb_init', None):
      pos_embedding = self.param('pos_embedding', cfg.posemb_init,
                                 pos_emb_shape)
    else:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(None, pos_emb_shape,
                                                           None)

    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: hyperparameters of the module
    out_dim: optionally specify out dimension.
  """
  config: ml_collections.ConfigDict
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs, train):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)
    x = nn.Dense(
        cfg.mlp_dim,
        dtype=cfg.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
    output = nn.Dense(
        actual_out_dim,
        dtype=cfg.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            x)
    output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=not train)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: hyperparameters of the module
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, train=False):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.
      train: whether to apply dropout

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=not train)(x, encoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MlpBlock(config=cfg)(y, train=train)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: hyperparameters of the module
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               decode=False,
               train=False):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
      decode: whether to run in single-position autoregressive mode.
      train: whether to apply dropout

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(targets)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=not train,
        decode=decode)(x, decoder_mask)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=not train)(y, encoded, encoder_decoder_mask)

    y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=not train)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(dtype=cfg.dtype)(y)
    z = MlpBlock(config=cfg)(z, train=train)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: hyperparameters of the module
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x,
               *,
               train: bool,
               debug: bool = False):
    """Applies Transformer model on the inputs."""
    cfg = self.config

    # Only spectrogram inputs are implemented for now.
    for modality in x:
      if modality == 'spectrogram':
        x_spec = x[modality]
      else:
        assert x[modality] is None

    x = []
    if 'spectrogram' in cfg.modality_fusion:
      x_spec, _ = temporal_encode(x_spec, 'spectrogram',
                                  cfg.temporal_encoding_config, cfg.patches,
                                  cfg.emb_dim)
      # TODO(valgab): Have different pos embeddings for different modalities
      x_spec = AddPositionEmbs(config=cfg, name='posembed_input')(x_spec)
      x.append(x_spec)
    x = jnp.concatenate(x, axis=1)

    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)

    x = x.astype(cfg.dtype)

    # Input Encoder
    encoder_mask = None
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(
          config=cfg, name=f'encoderblock_{lyr}')(
              x, encoder_mask, train=train)

    encoded = nn.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: hyperparameters of the module
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               decoder_mask=None,
               encoder_decoder_mask=None,
               decode=False,
               train=False):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
      decode: whether to run in single-position autoregressive mode.
      train: whether to apply dropout

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Output tokens embedding table
    output_embed = nn.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))

    y = targets.astype('int32')
    if not decode:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(
        config=cfg, name='posembed_output')(
            y, decode=decode)
    y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=not train)

    y = y.astype(cfg.dtype)

    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg, name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              decode=decode,
              train=train)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    if cfg.get('logits_via_embedding', True):
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          cfg.vocab_size,
          dtype=cfg.dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')(
              y)
    return logits


class Seq2SeqModule(nn.Module):
  """Transformer Model for sequence to sequence translation."""

  encoder_model: str
  encoder_config: ml_collections.ConfigDict
  decoder_model: str
  decoder_config: ml_collections.ConfigDict
  add_masked_word_prediction_loss: bool
  freeze_rgb_stream: bool
  dtype: jnp.dtype

  def setup(self):

    if self.encoder_model == 've':
      # Vanilla transformer encoder
      self.encoder = Encoder(config=self.encoder_config)
    elif self.encoder_model == 'mbt':
      self.encoder = mbt_model.MBT(
          num_classes=1,
          dtype=self.dtype,
          return_preclassifier=True,
          **self.encoder_config,
          name='video_encoder')
    self.decoder = Decoder(config=self.decoder_config)

  def encode(self,
             x_rgb: Optional[jnp.ndarray],
             x_flow: Optional[jnp.ndarray],
             x_spec: Optional[jnp.ndarray],
             x_wave: Optional[jnp.ndarray],
             x_text: Optional[jnp.ndarray],
             *,
             train: bool,
             debug: bool = False):
    """Applies Transformer encoder-branch on the inputs."""
    # TODO(valgab): Make attention masks for the case where input_segmentation
    # is not None
    x = {
        'rgb': x_rgb,
        'flow': x_flow,
        'spectrogram': x_spec,
        'wave': x_wave,
        'text': x_text
    }

    encoded = self.encoder(x, train=train, debug=debug)
    encoding_dict = None

    if self.freeze_rgb_stream:
      encoded['rgb'] = jax.lax.stop_gradient(encoded['rgb'])
      encoded = jnp.concatenate(
          [encoded[m] for m in self.encoder_config.modality_fusion], axis=1)
      logging.info('stop_gradient applied')
    elif self.add_masked_word_prediction_loss:
      encoding_dict = encoded
      encoded = jnp.concatenate(
          [encoded[m] for m in self.encoder_config.modality_fusion], axis=1)
    return encoded, encoding_dict

  def decode(
      self,
      encoded,
      targets,  # Used for teacher forcing
      decode: bool,
      train: bool,
      encoded_mask: Optional[jnp.ndarray] = None,
      debug: bool = False,
  ):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      targets: target data.
      decode: whether to run in single-position autoregressive mode.
      train: whether to apply dropout
      encoded_mask: mask tensor indicating valitity of each token in encoded.
      debug: debug mode

    Returns:
      logits array from transformer decoder.
    """
    cfg = self.decoder_config

    # Make padding attention masks.
    if decode:
      # for fast autoregressive decoding only a special encoder-decoder mask is
      # used.
      decoder_mask = None
    else:
      # Teacher forcing
      # No attention to target paddings, no attention to future tokens
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=cfg.dtype),
          nn.make_causal_mask(targets, dtype=self.dtype))
    encoder_decoder_mask = None
    if encoded_mask is not None:
      encoder_decoder_mask = encoded_mask[:, jnp.newaxis, jnp.newaxis, :]
    logits = self.decoder(
        encoded,
        targets,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        decode=decode,
        train=train)
    return logits.astype(self.dtype)

  def __call__(self,
               x_rgb: Optional[jnp.ndarray],
               x_flow: Optional[jnp.ndarray],
               x_spec: Optional[jnp.ndarray],
               x_wave: Optional[jnp.ndarray],
               x_text: Optional[jnp.ndarray],
               targets,
               masked_token_idxs: Optional[jnp.ndarray] = None,
               masked_token_idx_masks: Optional[jnp.ndarray] = None,
               masked_word_targets: Optional[jnp.ndarray] = None,
               decode: bool = False,
               *,
               train: bool,
               debug: bool = False):
    """Applies Transformer model on the inputs."""

    encoded = self.encode(
        x_rgb, x_flow, x_spec, x_wave, x_text, train=train, debug=debug)

    output = self.decode(encoded[0], targets, decode=decode, train=train)

    if not train or not self.add_masked_word_prediction_loss:
      return output

    assert masked_token_idxs is not None
    assert masked_token_idx_masks is not None
    assert masked_word_targets is not None
    assert encoded[1] is not None
    logging.info('encoded[0] %s', encoded[0])
    logging.info('encoded[1] %s', encoded[1])
    max_num_masked_words = masked_token_idxs.shape[1]
    x_out = []
    x_mask = []
    sample_masked_inputs = jax.vmap(
        jax.vmap(lambda x, y: x[y], (None, 0), 0), (0, 0), 0)
    for modality in self.encoder_config.modality_fusion:
      modality_feature = encoded[1][modality]
      if modality == 'spectrogram':
        logging.info('spectrogram feature %s', modality_feature)
        modality_feature_mask = masked_token_idx_masks
        if self.encoder_config.classifier == 'token':
          cls_token = sample_masked_inputs(
              modality_feature, jnp.zeros_like(masked_word_targets[..., 0:1]))
          modality_feature = modality_feature[:, 1:, :]
          cls_token_mask = jnp.ones_like(masked_token_idx_masks[..., 0:1])
        modality_feature = sample_masked_inputs(modality_feature,
                                                masked_token_idxs)
        if self.encoder_config.classifier == 'token':
          modality_feature = jnp.concatenate([cls_token, modality_feature], 2)
          modality_feature_mask = jnp.concatenate(
              [cls_token_mask, masked_token_idx_masks], 2)
        logging.info('spectrogram feature 2 %s', modality_feature)
      else:
        modality_feature = jnp.repeat(
            modality_feature[:, jnp.newaxis], max_num_masked_words, 1)
        modality_feature_mask = jnp.ones_like(modality_feature[..., 0])
      x_out.append(modality_feature)
      x_mask.append(modality_feature_mask)
    masked_input_features = jnp.concatenate(x_out, 2)
    masked_input_feature_masks = jnp.concatenate(x_mask, 2)
    logging.info('masked_input_features %s', masked_input_features)
    b, m, t, e = masked_input_features.shape
    masked_input_features = jnp.reshape(masked_input_features, [b * m, t, e])
    masked_input_masks = jnp.reshape(masked_input_feature_masks, [b * m, t])
    masked_word_targets = jnp.reshape(masked_word_targets, [b * m, -1])

    word_pred_output = self.decode(
        masked_input_features,
        masked_word_targets,
        decode=False,
        train=False,
        encoded_mask=masked_input_masks)

    return output, word_pred_output


class Seq2SeqModel(object):
  """Sequence to sequence model."""

  def __init__(
      self,
      config: Optional[ml_collections.ConfigDict],
      dataset_meta_data: Dict[str, Any],
  ) -> None:
    if config is None:
      logging.warning('You are creating the model with default config.')
      config = self.default_flax_model_config()
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.flax_model = self.build_flax_model()

  def build_flax_model(self) -> nn.Module:
    """Sequence to sequence flax module."""
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    encoder_model = self.config.model.get('encoder_model', 've')
    if encoder_model == 've':
      encoder_config = self.config.ve.model
    elif encoder_model == 'mbt':
      encoder_config = self.config.mbt.model
    decoder_model = self.config.model.get('decoder_model', 'vd')
    if decoder_model == 'vd':
      decoder_config = self.config.vd.model
    add_mwp = self.config.get('predict_masked_word', False)
    freeze_rgb_stream = self.config.model.get('freeze_rgb_stream', False)
    return Seq2SeqModule(
        dtype=model_dtype,
        encoder_model=encoder_model,
        encoder_config=encoder_config,
        decoder_model=decoder_model,
        decoder_config=decoder_config,
        add_masked_word_prediction_loss=add_mwp,
        freeze_rgb_stream=freeze_rgb_stream,)

  def get_metrics_fn(self, split: Optional[str] = None):
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      targets, weights)```
    """
    del split  # The metric function is the same for all splits.

    def metric_fn(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        weights: jnp.ndarray,
        target_is_onehot: bool = False,
        metrics: base_model.MetricNormalizerFnDict = _CLASSIFICATION_METRICS,
    ) -> Dict[str, Tuple[float, int]]:
      """Calcualte metrics for the classification task.


      Currently we assume each metric_fn has the API:
        ```metric_fn(logits, targets, weights)```
      and returns an array of shape [batch_size]. We also assume that to compute
      the aggregate metric, one should sum across all batches, then divide by
      the
      total samples seen. In this way we currently only support metrics of the
      1/N
      sum f(inputs, targets). Note, the caller is responsible for dividing by
      the normalizer when computing the mean of each metric.

      Args:
       logits: Output of model in shape [batch, length, num_classes].
       targets: Targets to be decoded.
       weights: Indicate which tokens are valid (1) vs padding (0).
       target_is_onehot: If the target is a one-hot vector.
       metrics: The classification metrics to evaluate. The key is the name of
         the metric, and the value is the metrics function.

      Returns:
        A dict of metrics, in which keys are metrics name and values are tuples
        of
        (metric, normalizer).
      """
      if target_is_onehot:
        one_hot_targets = targets
      else:
        one_hot_targets = common_utils.onehot(targets,
                                              logits.shape[-1])

      # This psum is required to correctly evaluate with multihost. Only host 0
      # will report the metrics, so we must aggregate across all hosts. The psum
      # will map an array of shape [n_devices, batch_size] -> [batch_size]
      # by summing across the devices dim. The outer sum then sums across the
      # batch dim. The result is then we have summed across all samples in the
      # sharded batch.
      evaluated_metrics = {}
      for key, val in metrics.items():
        evaluated_metrics[key] = model_utils.psum_metric_normalizer(  # pytype: disable=wrong-arg-types  # jax-ndarray
            (val[0](logits, one_hot_targets,  # pytype: disable=wrong-arg-types  # jax-types
                    weights), val[1](logits, one_hot_targets, weights)))  # pytype: disable=wrong-arg-types  # jax-types
      return evaluated_metrics  # pytype: disable=bad-return-type  # jax-types

    return metric_fn

  def loss_function(
      self,
      logits: jnp.ndarray,
      targets: jnp.ndarray,
      weights: jnp.ndarray,
      model_params: Optional[jnp.ndarray] = None,
  ) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      targets: Targets to be decoded.
      weights: Indicate which tokens are valid (1) vs padding (0).
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """

    if self.config.get('predict_masked_word', False):
      logits, masked_word_logits = logits
      targets, masked_word_targets = targets
      weights, masked_word_weights = weights

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = targets
    else:
      one_hot_targets = common_utils.onehot(targets, logits.shape[-1])

    sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
        logits,
        one_hot_targets,
        weights,
        label_smoothing=self.config.get('label_smoothing'))

    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss

    if self.config.get('predict_masked_word', False):
      mwp_loss = model_utils.weighted_softmax_cross_entropy(
          masked_word_logits,
          common_utils.onehot(masked_word_targets,
                              masked_word_logits.shape[-1]),
          masked_word_weights,
          label_smoothing=self.config.get('label_smoothing'))
      total_loss += mwp_loss * self.config.get('mwp_loss_factor', 1.0)

    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({})
