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

"""Vision Transformer with Adaptive Computation."""

from typing import Any, Optional

from absl import logging
import flax
import flax.linen as nn
from flax.training import common_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import nn_layers
from scenic.projects.adatape import layers as adatape_layers
from scenic.projects.baselines import vit


def ponder_loss_fn(loss_atr: jnp.ndarray,
                   weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Ponder Loss for UT.

  Args:
    loss_atr: Input array of any shape.
    weights: None or array of any shape.

  Returns:
    loss: A scaler to regularize the ACT
  """
  if weights is not None:
    normalization = weights.sum() + 1e-8
  else:
    normalization = np.prod(loss_atr.shape)
  loss = jnp.sum(loss_atr) / normalization
  return loss


class AdaTapeViT(nn.Module):
  """AdaTape Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    ac_config: Configuration of the adaptive computation.
    hidden_size: Size of the hidden state of the output of model's stem. if
      None, we skip the extra projection + tanh activation at the end.
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
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    fh, fw = self.patches.size
    x_input = x
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    # We use linear projection (large stride CNN) to encode patches, only when
    # we are using input-driven bank.
    if self.ac_config.bank_type == 'input':
      patch_bank_size = self.ac_config.patch_bank_size
      bank = nn.Conv(
          self.hidden_size, (patch_bank_size, patch_bank_size),
          strides=(patch_bank_size, patch_bank_size),
          padding='VALID',
          name='embedding_bank')(
              x_input)
      _, bank_h, bank_w, _ = bank.shape
      bank = jnp.reshape(bank, [n, bank_h * bank_w, c])
      bank = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input_bank')(
              bank)
      bank = nn.Dense(bank.shape[-1], name='pre_bank')(bank)
    else:
      bank = None
    x = jnp.reshape(x, [n, h * w, c])
    if -1 in self.ac_config.get('add_tape_token_to_layers', []):
      x = adatape_layers.AddTapeToken(ac_config=self.ac_config)(
          x, bank, train=train)

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    x, aux_output = adatape_layers.AdaTapeEncoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        ac_config=self.ac_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='Transformer')(
            x, bank, train=train)

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
    return x, aux_output


class AdaTapeParity(nn.Module):
  """AdaTape Transformer model for Parity task.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    ac_config: Configuration of the adaptive computation.
    hidden_size: Size of the hidden state of the output of model's stem. if
      None, we skip the extra projection + tanh activation at the end.
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
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    if self.ac_config.bank_type == 'input':
      x_input = x
      n, s, c = x.shape
      x = jnp.reshape(x, [n, 1, s * c])
      x = nn.Dense(self.hidden_size, name='embedding')(x)
      bank = nn.Dense(self.hidden_size, name='pre_bank')(x_input)
    else:
      raise NotImplementedError

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, self.hidden_size),
                       x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x, aux_output = adatape_layers.AdaTapeEncoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        ac_config=self.ac_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='Transformer')(
            x, bank, train=train)

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
    return x, aux_output


class AdaTapeMultiLabelClassificationModel(MultiLabelClassificationModel):
  """AdaTape Transformer model for multi-label classification task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    # For the parity task, we use AdaTapeParity as our model.
    if self.config.model_name == 'adatape-parity':
      return AdaTapeParity(
          num_classes=self.dataset_meta_data['num_classes'],
          mlp_dim=self.config.model.mlp_dim,
          num_layers=self.config.model.num_layers,
          num_heads=self.config.model.num_heads,
          patches=self.config.model.patches,
          ac_config=self.config.model.ac_config,
          hidden_size=self.config.model.hidden_size,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                       0.1),
          stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
          dtype=model_dtype,
      )
    else:
      return AdaTapeViT(
          num_classes=self.dataset_meta_data['num_classes'],
          mlp_dim=self.config.model.mlp_dim,
          num_layers=self.config.model.num_layers,
          num_heads=self.config.model.num_heads,
          patches=self.config.model.patches,
          ac_config=self.config.model.ac_config,
          hidden_size=self.config.model.hidden_size,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                       0.1),
          stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
          dtype=model_dtype,
      )

  def loss_function(
      self,
      logits: jnp.ndarray,
      batch: base_model.Batch,
      model_params: Optional[jnp.ndarray] = None,
      auxiliary_outputs: Any = None,
  ) -> float:
    """Returns sigmoid cross entropy loss with an L2 penalty (and ponder loss) on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.
      auxiliary_outputs: Output of model auxiliary_outputs.

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
    act_config = self.config.model.ac_config.get('dynamic_tape_length')
    if (act_config
        is not None) and (act_config.act_loss_weight > 0.0):
      ponder_loss = ponder_loss_fn(auxiliary_outputs[1], weights)
      total_loss += act_config.act_loss_weight * ponder_loss
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
    return initialise_from_train_state(train_state, restored_train_state,
                                       self.config, restored_model_cfg)


def initialise_from_train_state(
    train_state: Any, restored_train_state: Any,
    config: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> Any:
  """Updates the train_state with data from restored_train_state.

  Args:
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    config: Configurations for the model being updated.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.

  Returns:
    Updated train_state.
  """
  del config
  del restored_model_cfg
  # Create restored parameters dict:
  params = flax.core.unfreeze(train_state.optimizer.target)
  params_dict = {
      '/'.join([str(kk)
                for kk in k]): v
      for k, v in flax.traverse_util.flatten_dict(params).items()
  }
  # Create restored parameters dict:
  restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)
  restored_params_dict = dict()
  for key, value in flax.traverse_util.flatten_dict(restored_params).items():
    name = '/'.join([str(k) for k in key])
    restored_params_dict[name] = value
  # Copy parameters over:
  for pname, pvalue in restored_params_dict.items():
    if 'output_projection' in pname:
      continue
    elif 'pos_embedding' in pname:
      # TODO(dehghani) add support for reshaping pos-embedding to longer seq
      #  (e.g., for high res finetuning.).
      continue
    elif pname in params_dict:
      params_dict[pname] = pvalue
    else:
      logging.error("Restored key doesn't exist in the model: %s.", pname)
  logging.info('Inspect missing keys from the restored params:\n%s',
               params_dict.keys() - restored_params_dict.keys())
  logging.info('Inspect extra keys the the restored params:\n%s',
               restored_params_dict.keys() - params_dict.keys())
  # Restore data format
  splitkeys = {tuple(k.split('/')): v for k, v in params_dict.items()}
  params = flax.traverse_util.unflatten_dict(splitkeys)
  logging.info('Parameter summary after initialising from train state:')
  debug_utils.log_param_shapes(params)
  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))
