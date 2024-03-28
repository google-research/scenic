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

"""AdaTape Layers."""
from typing import Any, Callable, Optional, Sequence, Tuple

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.layers import nn_layers

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def truncated_normal_initializer():
  """TruncatedNormal(0.02) initializer from BERT."""

  def init(key, shape, dtype=jnp.float32):
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return jax.random.truncated_normal(key, -2, 2, shape, dtype) * 0.02

  return init


class AddTapeToken(nn.Module):
  """Adds tape token to the input."""
  ac_config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x: jnp.ndarray, bank: Optional[jnp.ndarray],
               train: bool) -> Any:
    """Add tape tokens."""
    feature_size = x.shape[-1]
    logging.info('Input shape before adding tape tokens: %s', x.shape)
    # Retrieve tape tokens from a tape bank:
    # For now, we use CLS token, but we can have a separate <TAPE> token.
    # 'token' or 'gap' to generate query.
    # We use the same type as ViT classifier.
    if self.ac_config.query_type == 'token':
      tape_token_query = x[:, 0]
    else:
      tape_token_query = jnp.mean(x, axis=1)
    tape_tokens, aux_output = TapeBank(
        ac_config=self.ac_config,
        features=feature_size,
        tape_init=truncated_normal_initializer())(
            tape_token_query, bank, not train)
    # Optionally apply an MLP with GLU activation
    if self.ac_config.get('tt_mlp_dim', 0):
      tape_tokens = MlpBlock(
          mlp_dim=self.ac_config.tt_mlp_dim,
          dropout_rate=self.ac_config.tt_dropout_rate,
          activation_fn=nn.glu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))(
              tape_tokens, deterministic=not train)

    # Concat tape tokens to the input.
    x = jnp.concatenate([x, tape_tokens], axis=1)

    logging.info('Input shape after adding tape tokens: %s', x.shape)
    return x, aux_output


class ACTFunction(nn.Module):
  """Adaptive Computation Time Function to help we use nn.scan on ACT."""

  ac_config: ml_collections.ConfigDict
  features: int
  tape_tokens: Any
  deterministic: bool = False

  def setup(self):

    # Init how many tokens we append at each time step.
    self.num_token_per_step = self.ac_config.dynamic_tape_length.num_token_per_step
    # Init max steps we need to ponder.
    self.max_steps = self.ac_config.num_tape_tokens // self.num_token_per_step
    # Init the threshold for act.
    self.threshold = self.ac_config.dynamic_tape_length.act_epsilon

  def act_step(self, x) -> Any:

    # Unpack the carry input.
    (query, halting_prob, remainders, n_updates, score_mask) = x

    # Init the still running mask.
    still_running = jnp.less(halting_prob, self.threshold).astype(jnp.float32)

    # Init the tape token keys.
    # Compute similarity for tape tokens and generate the topK index.
    if self.ac_config.bank_type == 'learn':
      tape_token_keys = self.tape_tokens[:, :self.features //
                                         self.ac_config.split_tt]
      scores = jnp.dot(query, tape_token_keys.T)
    elif self.ac_config.bank_type == 'input':
      tape_token_keys = self.tape_tokens[:, :, :self.features //
                                         self.ac_config.split_tt]
      tape_token_keys = jnp.transpose(tape_token_keys, [0, 2, 1])
      scores = jnp.matmul(jnp.expand_dims(query, 1), tape_token_keys)
      scores = jnp.squeeze(scores, axis=1)
    else:
      raise NotImplementedError

    topk_idndex_inpool = jax.lax.top_k(
        scores - score_mask * 1e+9,
        self.ac_config.num_tape_tokens // self.threshold)[1]
    # Select the weights from the scores and softmax.
    weights = jnp.take(scores, topk_idndex_inpool)
    weights = nn.softmax(weights / query.shape[-1]**0.5)
    # Compute the entropy or max value for loss function.
    if self.ac_config.dynamic_tape_length.act_loss_type == 'entropy':
      entropy = 1.0 - jnp.sum(weights**2, axis=-1)
    else:
      entropy = 1.0 - jnp.max(weights, axis=-1)

    # Init the new halted mask.
    sum_weights = jnp.sum(weights[:, :self.num_token_per_step], axis=-1)
    new_halted = jnp.greater_equal(halting_prob + sum_weights,
                                   self.threshold).astype(
                                       jnp.float32) * still_running

    # Update still running.
    still_running = still_running - new_halted

    # Update remainder.
    remainders = remainders + (new_halted + still_running) * entropy

    # Update halting_prob.
    halting_prob = halting_prob + sum_weights * still_running
    halting_prob += new_halted * (self.threshold - halting_prob)

    # Increment n_updates for all inputs which are still running.
    n_updates += still_running + new_halted

    # Take the new selected tokens from the token bank,
    # and merge them into single tape token.
    if self.ac_config.bank_type == 'learn':
      token_selected_wo_merge = jnp.take(
          self.tape_tokens, topk_idndex_inpool, axis=0)
    elif self.ac_config.bank_type == 'input':
      token_selected_wo_merge = jnp.take_along_axis(
          self.tape_tokens, jnp.expand_dims(topk_idndex_inpool, -1), axis=1)
    else:
      raise NotImplementedError

    token_selected = token_selected_wo_merge * jnp.expand_dims(weights, -1)
    token_selected = jnp.sum(token_selected, axis=-2, keepdims=True)

    # Update score_mask according to the new selected tokens.
    if self.ac_config.bank_type == 'learn':
      score_mask += jnp.sum(
          jax.nn.one_hot(topk_idndex_inpool, self.tape_tokens.shape[0]), axis=1)
    elif self.ac_config.bank_type == 'input':
      score_mask += jnp.sum(
          jax.nn.one_hot(topk_idndex_inpool, self.tape_tokens.shape[1]), axis=1)
    else:
      raise NotImplementedError

    # Update the query.
    token_selected_keys = token_selected[:, :, :self.features //
                                         self.ac_config.split_tt]

    # Different mode to update query.
    # If True, replace the query by avg of old query and tape keys;
    # If False, replace the query by the tape keys directly.
    if self.ac_config.dynamic_tape_length.complex_query:
      query = (query + jnp.mean(token_selected_keys, axis=1)) / 2.0
    else:
      query = jnp.mean(token_selected_keys, axis=1)

    return (query, halting_prob, remainders, n_updates,
            score_mask), token_selected

  # Define one stop function to decide the routing result.
  def stop_fn(self, inputs: Any) -> jnp.ndarray:
    # Returns True if all of halting probability >= 1-eps.
    _, halting_prob, _, _, *_ = inputs
    return jnp.all(halting_prob >= self.threshold)

  def take_a_step(self, x) -> Any:
    return self.act_step(x)

  def skip_a_step(self, x) -> Any:  # Shunt
    bs = x[0].shape[0]
    empty_tokens = jnp.zeros([bs, self.num_token_per_step, self.features])
    return x, empty_tokens

  @nn.compact
  def __call__(self, carry_in, _) -> Any:
    if self.is_mutable_collection('params'):  # Init-mode
      carry_out, scan_out = self.take_a_step(carry_in)
    else:
      decision = self.stop_fn(carry_in)
      carry_out, scan_out = nn.cond(decision, self.skip_a_step,
                                    self.take_a_step, self, carry_in)
    return carry_out, scan_out


class ATRTapeAppender(nn.Module):
  """ATRTapeToken Module.

  Given the TAPE token embedding, returns the top-k tpe tokens from the bank.

  Attributes:
    ac_config: ml_collections.ConfigDict to use adaptive config
    features: Number of feature dimensions for each embedding.
    tape_tokens: Tape initializer.
    dtype: The dtype of the embedding vectors (default: float32).
  """
  ac_config: ml_collections.ConfigDict
  features: int
  tape_tokens: Any
  num_tape_tokens: int

  def setup(self):

    # Init how many tokens we append at each time step.
    self.num_token_per_step = self.ac_config.dynamic_tape_length.num_token_per_step
    # Init max steps we need to ponder.
    self.max_steps = self.ac_config.num_tape_tokens // self.num_token_per_step

  @nn.compact
  def __call__(self, query: jnp.ndarray, deterministic: bool) -> jnp.ndarray:

    bs = query.shape[0]
    # Init the values for scan func.
    # Carry value required: query, halting_probability,
    # loss_atr, n_updates, score_mask.
    # Dynamic shape for update tensors below.
    update_shape = query.shape[:1]
    halting_probability = jnp.zeros(update_shape)
    # ATR loss term for each sample.
    loss_atr = jnp.zeros(update_shape)
    # Number of updates performed (N(t) in the paper).
    n_updates = jnp.zeros(update_shape)
    # Scan value required: None.
    score_mask = jnp.zeros([update_shape[0], self.tape_tokens.shape[-2]])

    # One trick to enhance AdaTape with Learnable bank.
    # We mask a subset of the whole bank during training.
    if not deterministic and self.ac_config.dynamic_tape_length.bernoulli_p > 0.0:
      # We split the rng, one for longer sequence,
      # another one is for early exit.
      rng = self.make_rng('dropout')
      _, rng = jax.random.split(rng)
      score_mask = score_mask + jax.random.bernoulli(
          rng,
          p=self.ac_config.dynamic_tape_length.bernoulli_p,
          shape=score_mask.shape).astype(jnp.float32)

    # Pack the carry input for nn.scan.
    carry_input = (query, halting_probability, loss_atr, n_updates, score_mask)

    # Init act_fn.
    act_fn = nn.scan(
        ACTFunction,
        variable_broadcast='params',
        split_rngs={
            'params': False,
            'dropout': True
        },
        length=self.max_steps)

    # Conduct the act_fn.
    carry_output, scan_out = act_fn(self.ac_config, self.features,
                                    self.tape_tokens,
                                    deterministic)(carry_input, None)

    # Reshapt the outputs of nn.scan.
    scan_out = jnp.transpose(scan_out, (1, 0, 2, 3))
    scan_out = jnp.reshape(
        scan_out, (bs, self.max_steps * self.num_token_per_step, self.features))
    _, halting_probability, remainders, n_updates, _ = carry_output

    return scan_out, (remainders, n_updates)  # pytype: disable=bad-return-type  # jax-ndarray


class TapeBank(nn.Module):
  """TapeBank Module.

  Given the TAPE token embedding, returns the top-k tpe tokens from the bank.

  Attributes:
    ac_config: Configuration of the adaptive computation.
    features: Number of feature dimensions for each embedding.
    tape_init: Tape initializer.
    dtype: The dtype of the embedding vectors (default: float32).
  """
  ac_config: ml_collections.ConfigDict
  features: int
  tape_init: Initializer
  dtype: jnp.ndarray = jnp.float32

  def setup(self):
    self.split_tt = self.ac_config.split_tt
    self.num_tape_tokens = self.ac_config.num_tape_tokens
    self.tape_bank_size = self.ac_config.tape_bank_size
    if self.ac_config.bank_type == 'learn':
      self.tape_tokens = self.param('tape_tokens', self.tape_init,
                                    (self.tape_bank_size, self.features),
                                    self.dtype)
    self.norm_layer = nn.LayerNorm(name='bank_norm', dtype=self.dtype)
    self.dy_config = self.ac_config.dynamic_tape_length

  @ nn.compact
  def __call__(self, query: jnp.ndarray,
               bank: Optional[jnp.ndarray], deterministic: bool) -> jnp.ndarray:
    """Retrieve tape tokens from a bank.

    Args:
      query: Array with last dimension equal the feature depth `features` of
        the embedding of tape tokens.
      bank: Array with candidate tape tokens.
      deterministic: bool denotes training or not.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    # One trick to improve AdaTape with learnable bank.
    # Add noise into to query during training.
    if not deterministic and self.dy_config and self.dy_config.query_noise > 0.0:
      rng = self.make_rng('dropout')
      _, rng = jax.random.split(rng)
      query += self.dy_config.query_noise * jax.random.normal(rng, query.shape)
    if self.ac_config.bank_type == 'learn':
      tape_tokens = jnp.asarray(self.tape_tokens, self.dtype)
    elif self.ac_config.bank_type == 'input':
      tape_tokens = bank
    else:
      raise NotImplementedError
    # Norm query and bank with the same LayerNorm.
    query = self.norm_layer(query)
    tape_tokens = self.norm_layer(tape_tokens)
    # Split the tape token [:,:, feature_dim] into two sub-vectors, i.e. key
    # and value. We also use half of the query as the real query if so.
    feature_dim = query.shape[-1]
    query = query[:, :feature_dim // self.split_tt]
    # We use ATR for dynamic reading.
    if self.dy_config:
      return ATRTapeAppender(self.ac_config, self.features, tape_tokens,
                             self.num_tape_tokens)(query, deterministic)
    # `scores` is an array with final dim `tape_bank_size` corresponding to the
    # batched inner-product of the array of query vectors against
    # each tape token in tape_tokens.

    # When we do not consider adaptive length, we use the following code.
    if self.ac_config.bank_type == 'learn':
      tape_tokens_keys = tape_tokens[:, :feature_dim // self.split_tt]
      scores = jnp.dot(query, tape_tokens_keys.T)
      topk_idndex = jax.lax.top_k(scores, self.num_tape_tokens)[1]
      assert jnp.issubdtype(topk_idndex.dtype, jnp.integer)
      return jnp.take(tape_tokens, topk_idndex, axis=0), (None, None)  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      token_selected_keys = tape_tokens[:, :, :self.features //
                                        self.ac_config.split_tt]
      token_selected_keys = jnp.transpose(token_selected_keys, [0, 2, 1])
      scores = jnp.matmul(jnp.expand_dims(query, 1), token_selected_keys)
      scores = jnp.squeeze(scores, axis=1)
      topk_idndex = jax.lax.top_k(scores, self.num_tape_tokens)[1]
      assert jnp.issubdtype(topk_idndex.dtype, jnp.integer)
      topk_idndex = jnp.expand_dims(topk_idndex, axis=-1)
      tape_tokens = jnp.take_along_axis(tape_tokens, topk_idndex, axis=1)
      return tape_tokens, (None, None)  # pytype: disable=bad-return-type  # jax-ndarray


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  ac_config: ml_collections.ConfigDict
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
               inputs_q: jnp.ndarray,
               inputs_kv: jnp.ndarray = None,
               input_mask: Optional[jnp.ndarray] = None,
               added_tape_len: int = 0,
               deterministic: bool = None) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs_q: Input data used to generate query.
      inputs_kv: Input data used to generate key/value.
      input_mask: Input mask, used for text input.
      added_tape_len: Length of the tape that is added to the original input, in
        terms of number of tape tokens.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    assert inputs_q.ndim == 3
    # Attention block.
    x_q = nn.LayerNorm(dtype=self.dtype)(inputs_q)
    if inputs_kv is not None:
      assert inputs_kv.ndim == 3
      x_kv = nn.LayerNorm(dtype=self.dtype)(inputs_kv)
    else:
      x_kv = x_q
    if input_mask is not None:
      attention_mask = input_mask[:, None, None, :] * jnp.ones(
          [1, 1, x_q.shape[1], 1])
    else:
      attention_mask = None

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(
            x_q, x_kv, mask=attention_mask)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs_q

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    use_tap_mlp = self.ac_config.get('enc_tt_mlp_dim', 0) and added_tape_len
    if use_tap_mlp:
      # Slice out y, and y_tapes:
      y, y_tapes = jnp.split(y, [(y.shape[1] - added_tape_len)], axis=1)
      y_tapes = MlpBlock(
          mlp_dim=self.ac_config.enc_tt_mlp_dim,
          dtype=self.dtype,
          dropout_rate=self.ac_config.enc_tt_dropout_rate,
          activation_fn=nn.glu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='tape_mlp')(
              y_tapes, deterministic=deterministic)

    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)
    if use_tap_mlp:
      y = jnp.concatenate([y, y_tapes], axis=1)

    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


def update_input_mask(
    input_mask: Optional[jnp.ndarray], taped_x: jnp.ndarray,
    ada_tt_len: Optional[jnp.ndarray], ac_config: ml_collections.ConfigDict
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
  """Updates input mask and taped_x if needed."""
  logging_input_mask = None
  bs, in_len, _ = taped_x.shape
  num_tape_tokens = ac_config.num_tape_tokens
  if ac_config.get('dynamic_tape_length'):
    if input_mask is None:
      # Set the mask to one for all the tokens from the original input.
      input_mask = jnp.ones((bs, (in_len - num_tape_tokens)))
    assert ada_tt_len is not None
    tape_mask = jnp.tile(jnp.arange(num_tape_tokens),
                         (bs, 1)) < ada_tt_len[..., None]
    input_mask = jnp.concatenate(
        [input_mask, tape_mask.astype(input_mask.dtype)], axis=1)
    logging_input_mask = input_mask
  elif (
      input_mask is not None
      # Only update the mask if its size doesn't match the current input,
      # which means we need mask to also cover added tape tokens.
      and in_len != input_mask.shape[1]):
    new_len = taped_x.shape[1]
    # Update the input mask to include tape tokens.
    tape_mask = jnp.ones((input_mask.shape[0], (new_len - input_mask.shape[1])))
    input_mask = jnp.concatenate([input_mask, tape_mask], axis=1)

  return input_mask, logging_input_mask, taped_x  # pytype: disable=bad-return-type  # jax-ndarray


def get_q_kv_mask(
    x: jnp.ndarray,
    input_mask: Optional[jnp.ndarray],
    layer: int,
    tape_added: int,
    ac_config: ml_collections.ConfigDict,
    bank: Optional[jnp.ndarray],
    train: bool,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray],
           Optional[jnp.ndarray], Optional[jnp.ndarray], int]:
  """Generates query, key/valye, input mask and logging input mask based on ac_config."""

  # Prepare x and taped_x, if necessary:
  if layer in ac_config.add_tape_token_to_layers:
    # For layers where we add tape token:
    taped_x, (loss_atr, n_updates) = AddTapeToken(ac_config=ac_config)(
        x, bank=bank, train=train)
    tape_added += taped_x.shape[1] - x.shape[1]
    # Correct n_updates when the num_token_per_step > 1.
    if ac_config.dynamic_tape_length:
      n_updates = n_updates * ac_config.dynamic_tape_length.num_token_per_step

    input_mask, logging_input_mask, taped_x = update_input_mask(
        input_mask=input_mask,
        taped_x=taped_x,
        ada_tt_len=n_updates,
        ac_config=ac_config)

    # Prepare query and key/value (memory):
    x_q, x_kv = taped_x, None

    return x_q, x_kv, input_mask, logging_input_mask, loss_atr, tape_added

  elif tape_added:
    # For layers after adding tape tokens, where we don't add tape token.
    #   taped_x = x
    #   x = x[:, :-tape_added, :]
    x_q, x_kv = x, None
    return x_q, x_kv, input_mask, input_mask, None, tape_added
  else:
    # For layers before adding tape tokens, just run self attention:
    return x, None, input_mask, input_mask, None, 0


class AdaTapeEncoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    ac_config: Configuration of the adaptive computation.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Attention dropout rate
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  ac_config: ml_collections.ConfigDict
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               bank: Optional[jnp.ndarray] = None,
               *,
               input_mask: Optional[jnp.ndarray] = None,
               train: bool = False):
    """Applies Transformer model on the inputs."""

    assert x.ndim == 3  # Shape is `[batch, len, emb]`.
    # Init it as None and update it later.
    logging_input_mask = None
    tape_added = 0
    loss_atr = None
    # Input Encoder.
    for lyr in range(self.num_layers):
      if self.ac_config.get('add_tape_token_to_layers', []):
        output_q_kv = get_q_kv_mask(
            x, input_mask, lyr, tape_added, self.ac_config, bank, train=train)
        (x_q, x_kv, input_mask, logging_input_mask_tmp, loss_atr_tmp,
         tape_added) = output_q_kv
        # Update logging_input_mask only when the returned value is not None.
        if logging_input_mask_tmp is not None:
          logging_input_mask = logging_input_mask_tmp
        # Update loss_atr only when the returned value is not None.
        if loss_atr_tmp is not None:
          loss_atr = loss_atr_tmp
      else:
        # Add no tape token
        x_q, x_kv = x, None

      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          ac_config=self.ac_config,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
          self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
              x_q,
              x_kv,
              input_mask=input_mask,
              added_tape_len=tape_added,
              deterministic=not train)

    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded, (logging_input_mask, loss_atr)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    dense_dim = self.mlp_dim * (2 if self.activation_fn == nn.glu else 1)
    x = nn.Dense(
        dense_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output
