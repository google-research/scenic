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

"""Model utils for ViViT."""

from typing import Any, Optional, Tuple

from absl import logging
import flax
from flax.linen import linear
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models import model_utils as base_model_utils
import scipy


def reshape_to_1d_factorized(x: jnp.ndarray, axis: int):
  """Converts 2d inputs to 1d for axial attention."""

  assert x.ndim == 4, ('The input dimention should be '
                       '[batch_size, height, width, channel]')
  batch_size, height, width, channel = x.shape
  if axis == 1:
    return x.transpose((0, 2, 1, 3)).reshape(batch_size * width, height,
                                             channel)
  elif axis == 2:
    return x.reshape(batch_size * height, width, channel)


def reshape_to_2d_factorized(x: jnp.ndarray, axis: int,
                             two_d_shape: Tuple[int, int, int, int]):
  """Converts 1d inputs back to 2d after axial attention."""
  assert x.ndim == 3, ('The input dimention should be '
                       '[batch_size, height*width, channel]')
  batch_size, height, width, channel = two_d_shape
  if axis == 1:
    assert x.shape[0] == batch_size * width
    return x.reshape((batch_size, width, height, channel)).transpose(
        (0, 2, 1, 3))
  elif axis == 2:
    assert x.shape[0] == batch_size * height
    return x.reshape(two_d_shape)


def factorized_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Any] = None,
    dropout_rate: float = 0.1,
    deterministic: bool = False,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[jax.lax.Precision] = None,
) -> jnp.ndarray:
  """Applies head-factorized qkv dot-product attention.

  This factorizes the dot-product attention by assigning different
  heads to run attention on different axes.


  Args:
    query: Queries for calculating attention with shape of `[batch...,
      num_heads, qk_depth_per_head]`.
    key: Keys for calculating attention with shape of `[batch..., num_heads,
      qk_depth_per_head]`.
    value: Values to be used in attention with shape of `[batch..., num_heads,
      v_depth_per_head]`.
    bias: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch...]`. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc. Default
        is None, which means no bias is applied on attention matrix.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey to be used for dropout.
    dropout_rate: Dropout rate.
    deterministic: Deterministic or not (to apply dropout).
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, ..., num_heads, features]`.
  """
  if query.shape != key.shape:
    raise ValueError('Axial dot product attention only supports '
                     'query and key with the same shape.')

  if bias is not None:
    raise ValueError('Bias is not supported in '
                     'factorized_dot_product_attention.')

  # Normalize the query with the square of its depth.
  query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
  # Shape of query, key, and value: [bs, t, hw, h, c].

  prefix_str = 'abcdefghijk'
  # Split heads for each axial attention dimension.
  num_attn_dimensions = query.ndim - 3  # all dims but bs, heads, and channel.
  if query.shape[-2] % num_attn_dimensions != 0:
    raise ValueError(f'In head-axial dot-product attention, number of '
                     f'heads ({query.shape[-2]}) should be divisible by number '
                     f'of attention dimensions ({num_attn_dimensions})!')

  queries = jnp.split(query, num_attn_dimensions, axis=-2)
  keys = jnp.split(key, num_attn_dimensions, axis=-2)
  values = jnp.split(value, num_attn_dimensions, axis=-2)
  # queries, keys, and values are each a list with two arrays (sinec
  # we have two dims, t and hw) that are made by spliting heads:
  # [(bs, t, hw, h//2, c), (bs, t, hw, h//2, c)].

  outputs = []
  for i, (query, key, value) in enumerate(zip(queries, keys, values)):
    # Shape of query, key, and value: [bs, t, hw, h//2, c].
    axis = i + 1  # to account for the batch dim
    batch_dims = prefix_str[:axis]
    einsum_str = f'{batch_dims}x...z,{batch_dims}y...z->{batch_dims}x...y'
    # For axis=1 einsum_str (q,k->a): ax...z,ay...z->ax...y
    # For axis=2 einsum_str (q,k->a): abx...z,aby...z->abx...y
    attn_logits = jnp.einsum(einsum_str, query, key, precision=precision)
    # For axis=1 (attention over t): attn_logits.shape: [bs, t, hw, h//2, t]
    # For axis=2 (attention over hw): attn_logits.shape: [bs, t, hw, h//2, hw]
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    # Apply dropout.
    if not deterministic and dropout_rate > 0.:
      if dropout_rng is None:
        raise ValueError('Did not provide `rng` to dot_product_attention().')
      keep_prob = 1.0 - dropout_rate
      if broadcast_dropout:
        # Dropout is broadcast across the batch+head+non-attention dimension.
        dropout_shape = list(attn_weights.shape)
        dropout_shape[0] = 1  # Broadcast batch.
        dropout_shape[-2] = 1  # Broadcast heads.
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      else:
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
      multiplier = (
          keep.astype(attn_weights.dtype) /
          jnp.asarray(keep_prob, dtype=attn_weights.dtype))
      attn_weights *= multiplier

    einsum_str = f'{batch_dims}x...y,{batch_dims}y...z->{batch_dims}x...z'
    # For axis=1 einsum_str (a,v->o): ax...y,ay...z->ax...z
    # For axis=2 einsum_str (a,v->o): abx...y,aby...z->abx...z
    outputs.append(
        jnp.einsum(einsum_str, attn_weights, value, precision=precision))

  # Output is list with two arrays [(bs, t, hw, h//2, c), (bs, t, hw, h//2, c)]
  # concatinate the heads.
  return jnp.concatenate(outputs, axis=-2)


def central_frame_initializer():
  """Initialisation function for 3D convolutional kernels.

  The filter is initialised such that it only depends on the input at the
  central (w.r.t the time dimension) frame.

  Returns:
    init: Initialisation function for Flax
  """

  def init(key, shape, dtype=jnp.float32):
    assert len(shape) == 5, ('Should be initialising 5-d kernels'
                             '(t, h, w, c_in, c_out')
    init_kernel = linear.default_kernel_init(key, shape, dtype)
    central_time_index = shape[0] // 2
    init_kernel = init_kernel.at[:, :, :central_time_index, :, :].set(0.0)
    init_kernel = init_kernel.at[:, :, central_time_index + 1:, :, :].set(0.0)

    return init_kernel

  return init


def average_frame_initializer():
  """Initialisation function for 3D convolutional kernels.

  The filter is initialised such that it applies the same weights on each
  frame of the input.
  This is similar to "filter inflation" in
    "Joao Carreira, and Andrew Zisserman.
     Quo vadis, action recognition? a new model and the kinetics dataset".
  However, "filter inflation" uses the filter weights from a pretrained 2D CNN,
  and replicates them over all time dimensions.

  Returns:
    init: Initialisation function for Flax
  """

  def init(key, shape, dtype=jnp.float32):
    logging.info('Initialising shape %s', shape)
    assert len(shape) == 5, ('Should be initialising 5-d kernels'
                             '(t, h, w, c_in, c_out')
    assert shape[0] > 1, 'Temporal dimension should be > 1'

    # Tiling the temporal dimension of a larger kernel ensures that the
    # normalisation is handled by default_kernel_init().
    init_kernel = linear.default_kernel_init(key, shape, dtype)
    init_kernel = jnp.tile(init_kernel[0:1, :, :, :, :],
                           [init_kernel.shape[0], 1, 1, 1, 1])

    return init_kernel

  return init


def interpolate_positional_embeddings(restored_posemb_grid, n_tokens):
  """Interpolate positional embeddings from one size to another.

  Args:
    restored_posemb_grid: Positional embeddings from restored model. Shape is
      [n_restored_tokens, d]. It is assumed that the restored model used square
      image patches.
    n_tokens: Number of tokens in the target model. It is assumed that the input
      patches and image of the target model are square.

  Returns:
    positional embedding resized to match n_tokens. Shape is [1, n_tokens, d]
  """

  restored_gs = int(np.sqrt(len(restored_posemb_grid)))
  gs = int(np.sqrt(n_tokens))
  logging.info('Resizing grid-size from %s to %s.', restored_gs, gs)
  restored_posemb_grid = restored_posemb_grid.reshape(restored_gs, restored_gs,
                                                      -1)
  zoom = (gs / restored_gs, gs / restored_gs, 1)
  restored_posemb_grid = scipy.ndimage.zoom(restored_posemb_grid, zoom, order=1)
  restored_posemb_grid = restored_posemb_grid.reshape(1, gs * gs, -1)
  return restored_posemb_grid


def tile_positional_embeddings(restored_posemb_grid, n_tokens):
  """Tile positional embeddings.

  Args:
    restored_posemb_grid: Positional embeddings from restored model. Shape is
      [n_restored_tokens, d]
    n_tokens: Number of tokens in the target model.

  Returns:
    positional embedding tiled to match n_tokens. Shape is [1, n_tokens, d]
  """

  num_repeats = int(n_tokens / len(restored_posemb_grid))
  logging.info('Tiling loaded positional embeddings (%d), %d times',
               len(restored_posemb_grid), num_repeats)
  restored_posemb_grid = np.concatenate(
      [restored_posemb_grid] * num_repeats, axis=0)
  restored_posemb_grid = np.expand_dims(restored_posemb_grid, axis=0)

  return restored_posemb_grid


def interpolate_1d_positional_embeddings(restored_posemb, n_tokens):
  """Interpolate one-dimensional positional embeddings.

  Used when the number of tokens at the input of the encoder is different
  between the pretrained and target models. This function is used for the
  temporal encoder in the Factorised Encoder model which has 1d positional
  embeddings.

  Args:
    restored_posemb: Positional embeddings from restored model. Shape is
      [n_restored_tokens, d].
    n_tokens: Number of tokens in the target model.

  Returns:
    positional embedding tiled to match n_tokens. Shape is [1, n_tokens, d].
  """

  zoom = (n_tokens / restored_posemb.shape[0], 1)
  logging.info('Interpolating embeddings by a factor of %s', zoom)
  restored_posemb = scipy.ndimage.zoom(restored_posemb, zoom, order=1)
  restored_posemb = np.expand_dims(restored_posemb, axis=0)

  return restored_posemb


def initialise_from_train_state(
    config,
    train_state: Any,
    restored_train_state: Any,
    restored_model_cfg: ml_collections.ConfigDict,
    restore_output_proj: bool,
    vivit_transformer_key: str = 'Transformer',
    log_initialised_param_shapes: bool = True) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    config: Configurations for the model being updated.
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restore_output_proj: If true, load the final output projection. Set
      to False if finetuning to a new dataset.
    vivit_transformer_key: The key used for storing the subtree in the
      parameters that keeps Transformer weights, that are supposed to be
      initialized from the given pre-trained model.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.

  Returns:
    Updated train_state.
  """
  if hasattr(train_state, 'optimizer'):
    # Inspect and compare the parameters of the model with the init-model.
    params = flax.core.unfreeze(train_state.optimizer.target)
    train_state_keys = train_state.optimizer.target.keys()
  else:
    params = flax.core.unfreeze(train_state.params)
    train_state_keys = train_state.params.keys()
  if hasattr(restored_train_state, 'optimizer'):
    if config.init_from.get('checkpoint_format', 'scenic') == 'big_vision':
      restored_params = restored_train_state.optimizer['target']
    else:
      restored_params = restored_train_state.optimizer.target
    restored_params = flax.core.unfreeze(restored_params)
  else:
    restored_params = flax.core.unfreeze(restored_train_state.params)

  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      if restore_output_proj:
        params[m_key] = m_params
      else:
        pass

    elif m_key == 'pre_logits':
      if config.model.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   if from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.model.representation_size
        params[m_key] = m_params

    elif m_key in {'Transformer', 'SpatialTransformer', 'TemporalTransformer'}:
      key_to_load = vivit_transformer_key
      is_temporal = False
      if m_key == 'TemporalTransformer':
        key_to_load = m_key
        is_temporal = True
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          init_posemb(params[key_to_load], m_params, config, restored_model_cfg,
                      is_temporal=is_temporal)
        elif 'encoderblock' in tm_key:
          init_encoderblock(params[key_to_load], m_params, tm_key,
                            config)
        else:  # Other parameters of the Transformer encoder.
          params[key_to_load][tm_key] = tm_params

    elif m_key == 'embedding':
      init_embedding(params, m_params, config)
    else:
      if m_key in train_state_keys:
        params[m_key] = m_params
      else:
        logging.info('Skipping %s. In restored model but not in target', m_key)

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  if hasattr(train_state, 'optimizer'):
    return train_state.replace(
        optimizer=train_state.optimizer.replace(
            target=flax.core.freeze(params)))
  else:
    return train_state.replace(params=flax.core.freeze(params))


def init_posemb(to_params,
                from_params,
                config,
                restored_model_cfg,
                is_temporal,
                posemb_name='posembed_input',
                restored_posemb_name='posembed_input'):
  """Initialize the positional embeddings."""
  if config.init_from.get('restore_positional_embedding', True):
    posemb = to_params[posemb_name]['pos_embedding']
    restored_posemb = from_params[restored_posemb_name]['pos_embedding']
    if restored_posemb.shape != posemb.shape:
      # Rescale the grid of pos, embeddings.
      # Default parameter shape is (1, N, 768)
      logging.info('Adapting positional embeddings from %s to %s',
                   restored_posemb.shape, posemb.shape)
      ntok = posemb.shape[1]
      if restored_model_cfg.model.classifier == 'token':
        # The first token is the CLS token.
        restored_posemb_grid = restored_posemb[0, 1:]
        if config.model.classifier == 'token':
          # CLS token in restored model and in target.
          cls_tok = restored_posemb[:, :1]
          ntok -= 1
        else:
          # CLS token in restored model, but not target.
          cls_tok = restored_posemb[:, :0]
      else:
        restored_posemb_grid = restored_posemb[0]
        if config.model.classifier == 'token':
          # CLS token in target, but not restored model.
          cls_tok = posemb[:, :1]
          ntok -= 1
        else:
          # CLS token not in target or restored model.
          cls_tok = restored_posemb[:, :0]
      if ((config.model.classifier == 'token') !=
          (restored_model_cfg.model.classifier == 'token')):
        logging.warning('Only one of target and restored model uses a '
                        'classification token.')

      if len(restored_posemb_grid) != ntok:  # We need a resolution change.
        if is_temporal:
          restored_posemb_grid = interpolate_1d_positional_embeddings(
              restored_posemb_grid, ntok)

        elif config.init_from.positional_embed_size_change == 'resize':
          restored_posemb_grid = interpolate_positional_embeddings(
              restored_posemb_grid, ntok)

        elif config.init_from.positional_embed_size_change == 'tile':
          restored_posemb_grid = tile_positional_embeddings(
              restored_posemb_grid, ntok)

        elif config.init_from.positional_embed_size_change == 'resize_tile':
          temp_encoding = config.model.temporal_encoding_config
          if temp_encoding.method == 'temporal_sampling':
            tokens_per_frame = int(ntok / temp_encoding.n_sampled_frames)
          elif temp_encoding.method == '3d_conv':
            n_frames = (
                config.dataset_configs.num_frames //
                config.model.patches.size[2])
            tokens_per_frame = ntok // n_frames
          else:
            raise AssertionError(
                f'Unknown temporal encoding {temp_encoding.method}')
          restored_posemb_grid = interpolate_positional_embeddings(
              restored_posemb_grid, tokens_per_frame)
          restored_posemb_grid = restored_posemb_grid[0]
          restored_posemb_grid = tile_positional_embeddings(
              restored_posemb_grid, ntok)

        else:
          raise AssertionError(
              'Unknown positional embedding size changing method')
      else:  # Sequence lengths are the same.
        # Adds batch dimension.
        restored_posemb_grid = restored_posemb_grid[None, ...]

      # Attach the CLS token again.
      if config.model.classifier == 'token':
        restored_posemb = jnp.array(
            np.concatenate([cls_tok, restored_posemb_grid], axis=1))
      else:
        restored_posemb = restored_posemb_grid

    to_params[posemb_name]['pos_embedding'] = restored_posemb
  else:
    logging.info('Not restoring positional encodings from pretrained model')


def init_encoderblock(to_params, from_params, tm_key, config):
  """Initialize encoder_block_parameters."""
  # Explicitly enumerate over the keys in the encoder-block. Don't just
  # assign the dictionary. It is possible for the target model to
  # contain keys that are not in the restored model.
  attention_type = config.model.attention_config.type
  for enc_key in from_params[tm_key].keys():
    if attention_type in [
        'spacetime', 'factorized_encoder', 'factorized_dot_product_attention'
    ]:
      assert enc_key in to_params[tm_key], '%s not in to_params[%s]' % (enc_key,
                                                                        tm_key)
      to_params[tm_key][enc_key] = from_params[tm_key][enc_key]

    elif attention_type == 'factorized_transformer_block':
      if config.init_from.get('init_spatial_transformer', True):
        to_params[tm_key]['encoderblock_space'] = from_params
      if config.init_from.get('init_temporal_transformer', True):
        to_params[tm_key]['encoderblock_time'] = from_params

    elif attention_type == 'factorized_self_attention_block':
      if enc_key in to_params[tm_key]:
        # We have an exact match. This would happen when loading weights from
        # another factorised encoder model.
        to_params[tm_key][enc_key] = from_params[tm_key][enc_key]
        logging.info('%s: Initialising %s directly from restored model', tm_key,
                     enc_key)
      elif enc_key == 'MultiHeadDotProductAttention_0':
        if config.init_from.get('init_spatial_transformer', True):
          logging.info(
              '%s: Initialising spatial transformer from '
              'pretrained weights', tm_key)
          to_params[tm_key]['MultiHeadDotProductAttention_space'] = from_params[
              tm_key][enc_key].copy()
        if config.init_from.get('init_temporal_transformer', True):
          logging.info(
              '%s: Initialising temporal transformer from '
              'pretrained weights', tm_key)
          to_params[tm_key]['MultiHeadDotProductAttention_time'] = from_params[
              tm_key][enc_key].copy()
      elif enc_key == 'LayerNorm_0':
        to_params[tm_key]['LayerNorm_space'] = from_params[tm_key][enc_key]
        if config.init_from.get('init_temporal_layer_norm', False):
          logging.info(
              '%s: %s Initialising temporal layer norm from '
              'restored model', tm_key, enc_key)
      # The following part could be made more generic.
      elif enc_key == 'LayerNorm_1':
        to_params[tm_key]['LayerNorm_mlp'] = from_params[tm_key][enc_key]
      elif enc_key == 'MlpBlock_0':
        to_params[tm_key]['MlpBlock'] = from_params[tm_key][enc_key]
      else:
        logging.info(
            'Key "%s" in restored model\'s encoder block not in '
            'target model', enc_key)
    else:
      raise ValueError(f'Unknown attention type {attention_type}')


def init_embedding(to_params, from_params, config):
  """Initialize input embedding."""
  if config.init_from.get('restore_input_embedding', True):
    input_kernel = to_params['embedding']['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']

    if input_kernel.shape != restored_kernel.shape:
      # Kernel dimensions are [t, h, w, c_in, c_out].
      assert config.model.temporal_encoding_config.method == '3d_conv', (
          'Input kernel dimensions should only differ if 3d_conv is the'
          'temporal encoding method')
      assert input_kernel.shape[1:] == restored_kernel.shape, (
          'All filter dimensions besides the temporal dimension should be'
          'equal. {} vs {}'.format(input_kernel.shape, restored_kernel.shape))

      kernel_init_method = (
          config.model.temporal_encoding_config.kernel_init_method
      )
      if kernel_init_method == 'average_frame_initializer':
        # This corresponds to "filter inflation" in
        # J Carreira and A Zisserman. Quo vadis, action recognition?
        # A new model and the kinetics dataset. CVPR 2017".
        logging.info('Initializing input kernel with filter inflation.')
        t = input_kernel.shape[0]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
        restored_kernel = np.tile(restored_kernel, [t, 1, 1, 1, 1]) / t
      elif kernel_init_method == 'average_arp_frame_initializer':
        # This corresponds to a combination of filter inflation and
        # the approximate rank pooling described in
        # H Bilen et al. Action Recognition with Dynamic Image Networks.
        # PAMI 2017.
        logging.info('Initialzing input kernel with ARP inflation')
        t = input_kernel.shape[0]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
        restored_kernel = np.tile(restored_kernel, [t, 1, 1, 1, 1])

        def average_arp(length):
          # Implements Equation 3 of Bilen et al. PAMI 2017.
          array = np.arange(1, length + 1)

          harmonic = np.zeros((length + 1))
          harmonic[1:] = np.cumsum(1.0 / array)

          array = 2 * (length - array + 1) - (length + 1) * (
              harmonic[-1] - harmonic[:-1])
          return array

        normalizer = average_arp(t) / t
        normalizer = np.reshape(normalizer, [t, 1, 1, 1, 1])
        restored_kernel = restored_kernel * normalizer
      elif kernel_init_method == 'central_frame_initializer':
        logging.info('Initializing input kernel to select centre frame.')
        central_time_index = input_kernel.shape[0] // 2
        temp = np.zeros(input_kernel.shape)
        temp[central_time_index] = restored_kernel.copy()
        restored_kernel = temp
      else:
        raise AssertionError(
            'Unknown input kernel initialization {}'.format(kernel_init_method))

    to_params['embedding']['kernel'] = restored_kernel
    to_params['embedding']['bias'] = restored_bias
  else:
    logging.info('Not restoring input embedding parameters')


def get_joint_logits_labels(logits, one_hot_targets, class_splits):
  """Returns joint pairs of logits and labels.

  Args:
    logits: Tensor of shape [n, c]
    one_hot_targets: Tensor of shape [n, c]
    class_splits: List of length 2. The two elements, c1 and c. Used in
      jnp.split. Size of the two splits is therefore c1 and (c - c1)

  Returns:
    pairwise_logits: Tensor of shape [n, c1 * c2]
    pairwise_labels: One-hot tensor of shape [n, c1 * c2]
  """

  assert len(class_splits) == 2, 'Class_splits should have length 2'
  assert logits.ndim == 2, 'Logits should have dimension of 2'
  assert one_hot_targets.ndim == 2, 'One hot target should have dimension of 2'

  n = logits.shape[0]

  logits_a, logits_b = jnp.split(logits, class_splits, axis=-1)[:-1]
  one_hot_a, one_hot_b = jnp.split(one_hot_targets, class_splits, axis=-1)[:-1]
  n_class_a, n_class_b = logits_a.shape[1], logits_b.shape[1]

  logits_a = jax.nn.softmax(logits_a, axis=-1)
  logits_b = jax.nn.softmax(logits_b, axis=-1)

  pairwise_logits = logits_a[:, :, jnp.newaxis] * logits_b[:, jnp.newaxis, :]
  pairwise_logits = jnp.reshape(pairwise_logits, [n, n_class_a * n_class_b])
  labels_a = jnp.argmax(one_hot_a, axis=-1)
  labels_b = jnp.argmax(one_hot_b, axis=-1)

  pairwise_labels = labels_a * n_class_b + labels_b
  pairwise_labels = common_utils.onehot(pairwise_labels, n_class_a * n_class_b)

  return pairwise_logits, pairwise_labels


def joint_accuracy(logits, one_hot_target, class_splits, weights=None):
  """Compute accuracy where both targets must be predicted correctly."""

  pairwise_logits, pairwise_labels = get_joint_logits_labels(
      logits, one_hot_target, class_splits)
  return base_model_utils.weighted_correctly_classified(pairwise_logits,
                                                        pairwise_labels,
                                                        weights)


def joint_top_k(logits, one_hot_target, class_splits, k=5, weights=None):
  """Compute top-k where both targets must be predicted correctly."""

  pairwise_logits, pairwise_labels = get_joint_logits_labels(
      logits, one_hot_target, class_splits)
  return base_model_utils.weighted_topk_correctly_classified(
      pairwise_logits, pairwise_labels, weights, k)


def adapt_old_configs(
    hparams: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Updates old configs with new namings."""
  with hparams.unlocked():
    # Make sure attention_config exists.
    attention_config = hparams.model.get('attention_config', None)
    if attention_config is None:
      hparams.model.attention_config = ml_collections.ConfigDict()
    att_type = hparams.model.attention_config.get('type', None)

    # Default ViViT.
    if att_type is None:
      hparams.model.attention_config.type = 'spacetime'

    # Handle V3.
    elif att_type == 'factorised_space_time':
      hparams.model.attention_config.type = 'factorized_self_attention_block'

    # Handle V1.
    if hparams.get('model_variant', 'vivit') == 'space_time_vivit':
      hparams.model.attention_config.type = 'factorized_encoder'

  return hparams
