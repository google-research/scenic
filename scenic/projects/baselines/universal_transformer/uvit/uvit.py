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

"""Universal Vision Transformer with Adaptive Computation Time."""

from typing import Any, Optional

import flax.linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.baselines.universal_transformer import layers


def ponder_loss_fn(
    ponder_times: jnp.ndarray,
    remainders: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Ponder Loss for UT.

  Args:
    ponder_times: Input array of any shape.
    remainders: Input array of any shape.
    weights: None or array of any shape.

  Returns:
    loss: A scaler to regularize the ACT
  """

  if weights is not None:
    normalization = weights.sum() + 1e-8
  else:
    normalization = np.prod(ponder_times.shape)

  p_t = ponder_times + remainders
  # We only do mean when ac_config.act_level == per_example.
  # When ac_config.act_level == per_token, we do sum first.
  if jnp.ndim(p_t) == 1:
    loss = jnp.sum(p_t) / normalization
  else:
    loss = jnp.sum(jnp.sum(p_t, axis=-1)) / normalization
  return loss


class UTStochasticDepth(nn.Module):
  """Performs layer-dropout (also known as stochastic depth).

  Described in
  Huang & Sun et al, "Deep Networks with Stochastic Depth", 2016
  https://arxiv.org/abs/1603.09382

  Attributes:
    rate: the layer dropout probability (_not_ the keep rate!).
    deterministic: If false (e.g. in training) the inputs are scaled by `1 / (1
      - rate)` and the layer dropout is applied, whereas if true (e.g. in
      evaluation), no stochastic depth is applied and the inputs are returned as
      is.
  Note: This is a repeated implementation of model_lib.nn_layers.StochasticDepth
    The implementation here is to match the nn.cond in UT
  """
  rate: float = 0.0
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               deterministic: Optional[bool] = None) -> jnp.ndarray:
    """Applies a stochastic depth mask to the inputs.

    Args:
      x: Input tensor.
      deterministic: If false (e.g. in training) the inputs are scaled by `1 /
        (1 - rate)` and the layer dropout is applied, whereas if true (e.g. in
        evaluation), no stochastic depth is applied and the inputs are returned
        as is.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    if self.rate <= 0.0:
      return x
    if deterministic:
      return x
    else:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      rng = self.make_rng('dropout')
      mask = jax.random.bernoulli(rng, self.rate, shape)
      return x * (1.0 - mask)


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
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  deterministic: bool = False

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=self.deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    x = UTStochasticDepth(rate=self.stochastic_depth)(x, self.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=self.deterministic)
    y = UTStochasticDepth(rate=self.stochastic_depth)(y, self.deterministic)
    return y + x


class UTEncoder(nn.Module):
  """Universal Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  parameter_sharing: bool = True
  ac_config: Optional[ml_collections.ConfigDict] = None
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    # We use layers.AdaptiveComputationTime only when we are doing ACT.
    if self.ac_config is None:
      # We make the layer first if we are using parameter sharing.
      if not self.parameter_sharing:
        for i in range(self.num_layers):
          x = Encoder1DBlock(
              mlp_dim=self.mlp_dim,
              num_heads=self.num_heads,
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=self.stochastic_depth,
              deterministic=not train,
              name='encoderblock_' + str(i),
              dtype=dtype)(
                  x)
      else:
        encoder_block = Encoder1DBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            deterministic=not train,
            name='encoderblock',
            dtype=dtype)
        for i in range(self.num_layers):
          x = encoder_block(x)
      auxiliary_outputs = None
    else:
      encoder_block = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=self.stochastic_depth,
          deterministic=not train,
          name='encoderblock',
          dtype=dtype)
      x, auxiliary_outputs = layers.AdaptiveComputationTime(
          self.ac_config, encoder_block, self.parameter_sharing,
          name='act')(x)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded, auxiliary_outputs


class UViT(nn.Module):
  """Universall Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    ac_config: Configuration of the adaptive computation.
    hidden_size: Size of the hidden state of the output of model's stem.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  ac_config: ml_collections.ConfigDict
  hidden_size: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  parameter_sharing: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    fh, fw = self.patches.size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x, auxiliary_outputs = UTEncoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        ac_config=self.ac_config,
        stochastic_depth=self.stochastic_depth,
        parameter_sharing=self.parameter_sharing,
        dtype=self.dtype,
        name='UTransformer')(
            x, train=train)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)

    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x, auxiliary_outputs


class UViTMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Universal Vision Transformer model for multi-label classification task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return UViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        ac_config=self.config.model.get('ac_config'),
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        parameter_sharing=self.config.model.get('parameter_sharing', True),
        dtype=model_dtype,
    )

  def loss_function(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      logits: jnp.ndarray,
      auxiliary_outputs: Any,
      batch: base_model.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> float:
    """Returns sigmoid cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      auxiliary_outputs: Output of model auxiliary_outputs, (ponder_times,
        remainders)
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')

    if self.dataset_meta_data.get('target_is_onehot', False):
      multihot_target = batch['label']
    else:
      # This is to support running a multi-label classification model on
      # single-label classification tasks
      multihot_target = common_utils.onehot(batch['label'], logits.shape[-1])

    sig_ce_loss = model_utils.weighted_sigmoid_cross_entropy(
        logits,
        multihot_target,
        weights,
        label_smoothing=self.config.get('label_smoothing'))
    if self.config.get('l2_decay_factor') is None:
      total_loss = sig_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = sig_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    ac_config = self.config.model.get('ac_config')
    if (ac_config is not None) and (ac_config.act_loss_weight > 0.0):
      ponder_loss = ponder_loss_fn(auxiliary_outputs[0], auxiliary_outputs[1])
      total_loss += ac_config.act_loss_weight * ponder_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    raise NotImplementedError
