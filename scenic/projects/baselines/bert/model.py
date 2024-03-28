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

"""BERT Model."""

from typing import Any, Dict

from absl import logging
import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.base_models.regression_model import RegressionModel
from scenic.projects.baselines.bert import bert_base_model
from scenic.projects.baselines.bert import layers


class BERT(nn.Module):
  """BERT."""

  stem_config: ml_collections.ConfigDict
  encoder_config: ml_collections.ConfigDict
  head_config: ml_collections.ConfigDict
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: Dict[str, jnp.ndarray],
               *,
               train: bool,
               transfer_mode=False,
               debug: bool = False):

    x, word_embeddings = layers.Stem(
        vocab_size=self.stem_config.vocab_size,
        type_vocab_size=self.stem_config.type_vocab_size,
        hidden_size=self.stem_config.hidden_size,
        max_position_embeddings=self.stem_config.max_position_embeddings,
        dropout_rate=self.stem_config.dropout_rate,
        embedding_width=self.stem_config.get('embedding_width'),
        dtype=self.dtype,
        name='stem')(
            input_word_ids=inputs['input_word_ids'],
            input_type_ids=inputs['input_type_ids'],
            input_mask=inputs['input_mask'],
            train=train)

    x = layers.BERTEncoder(
        mlp_dim=self.encoder_config.mlp_dim,
        num_layers=self.encoder_config.num_layers,
        num_heads=self.encoder_config.num_heads,
        dropout_rate=self.encoder_config.dropout_rate,
        attention_dropout_rate=self.encoder_config.attention_dropout_rate,
        pre_norm=self.encoder_config.pre_norm,
        dtype=self.dtype,
        name='bert_encoder')(
            x, input_mask=inputs['input_mask'], train=train)

    if self.head_config.type == 'pretraining':
      next_sentence_prediction_logits = layers.ClassificationHead(
          num_outputs=2,
          hidden_sizes=(x.shape[-1], self.head_config.hidden_size),
          nonlinearity=nn.tanh,
          dtype=self.dtype,
          name='next_sentence_prediction_head')(
              x, train=train)
      if transfer_mode:
        # Next sentence prediction head is a classification head and we can
        # reuse it for transfer evaluation on classification tasks.
        return next_sentence_prediction_logits

      masked_language_modeling_logits = layers.MaskedLanguageModelHead(
          dtype=self.dtype, name='masked_language_model_head')(
              x, inputs['masked_lm_positions'], word_embeddings, train=train)
      return {
          'nsp_logits': next_sentence_prediction_logits,
          'mlm_logits': masked_language_modeling_logits
      }
    elif self.head_config.type == 'classification':
      return layers.ClassificationHead(
          num_outputs=self.head_config.num_classes,
          hidden_sizes=(x.shape[-1], self.head_config.hidden_size),
          nonlinearity=nn.tanh,
          dtype=self.dtype,
          name='classification_head')(
              x, train=train)
    elif self.head_config.type == 'regression':
      return layers.ClassificationHead(
          num_outputs=1,
          hidden_sizes=(x.shape[-1], self.head_config.hidden_size),
          nonlinearity=nn.tanh,
          dtype=self.dtype,
          name='regression_head')(
              x, train=train)


class BERTModel(bert_base_model.BERTBaseModel):
  """BERT model."""

  def build_flax_model(self):
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    with self.config.unlocked():
      # Add vocabulary information from dataset meta-data to configs:
      self.config.model.stem.vocab_size = self.dataset_meta_data['vocab_size']
      self.config.model.stem.type_vocab_size = self.dataset_meta_data[
          'type_vocab_size']
    return BERT(
        stem_config=self.config.model.stem,
        encoder_config=self.config.model.encoder,
        head_config=self.config.model.head,
        dtype=model_dtype,
    )

  def init_from_bert_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state."""
    raise NotImplementedError


class BERTClassificationModel(ClassificationModel):
  """BERT Classification model."""

  def build_flax_model(self):
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    with self.config.unlocked():
      # Add vocabulary information from dataset meta-data to configs:
      self.config.model.stem.vocab_size = self.dataset_meta_data['vocab_size']
      self.config.model.stem.type_vocab_size = self.dataset_meta_data[
          'type_vocab_size']
      self.config.model.head.num_classes = self.dataset_meta_data['num_classes']
    return BERT(
        stem_config=self.config.model.stem,
        encoder_config=self.config.model.encoder,
        head_config=self.config.model.head,
        dtype=model_dtype,
    )

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state."""
    return init_bert_from_train_state(train_state, restored_train_state,
                                      self.config, restored_model_cfg)


class BERTRegressionModel(RegressionModel):
  """BERT Regression model."""

  def build_flax_model(self):
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    with self.config.unlocked():
      # Add vocabulary information from dataset meta-data to configs:
      self.config.model.stem.vocab_size = self.dataset_meta_data['vocab_size']
      self.config.model.stem.type_vocab_size = self.dataset_meta_data[
          'type_vocab_size']
    return BERT(
        stem_config=self.config.model.stem,
        encoder_config=self.config.model.encoder,
        head_config=self.config.model.head,
        dtype=model_dtype,
    )

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state."""
    return init_bert_from_train_state(train_state, restored_train_state,
                                      self.config, restored_model_cfg)


def init_bert_from_train_state(
    train_state: Any, restored_train_state: Any,
    config: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> Any:
  """Updates the train_state with data from restored_train_state."""
  del restored_model_cfg

  def _get_param_dict(params):
    return {
        '/'.join([str(kk)
                  for kk in k]): v
        for k, v in flax.traverse_util.flatten_dict(params).items()
    }

  if hasattr(train_state, 'optimizer'):
    # TODO(dehghani): Remove support for flax optim.
    params = flax.core.unfreeze(train_state.optimizer.target)
    restored_params = flax.core.unfreeze(
        restored_train_state.optimizer.target)
  else:
    params = flax.core.unfreeze(train_state.params)
    restored_params = flax.core.unfreeze(restored_train_state.params)

  params_dict = _get_param_dict(params)
  # Fix some names:
  restored_params_dict = dict()
  for key, value in flax.traverse_util.flatten_dict(restored_params).items():
    name = '/'.join([str(k) for k in key])
    if config.init_from.restore_next_sentence_prediction_head_params:
      name = name.replace('next_sentence_prediction_head',
                          'classification_head')
    restored_params_dict[name] = value
  # Copy parameters over:
  for pname, pvalue in restored_params_dict.items():
    if 'masked_language_model_head' in pname:
      # We throw away parameters of `masked_language_model_head`, but
      # for the `next_sentence_prediction_head`, we only discard the final
      # dense (`output_projection`) tha maps model representation to the
      # label space.
      continue
    if (not config.init_from.restore_next_sentence_prediction_head_params and
        'next_sentence_prediction_head' in pname):
      continue
    if 'output_projection' in pname:
      continue
    elif pname in params_dict:
      params_dict[pname] = pvalue
    else:
      logging.error("Restored key doesn't exist in the model: %s.", pname)

  logging.info('Inspect missing keys from the restored params:\n%s',
               params_dict.keys() - restored_params_dict.keys())
  logging.info('Inspect extra keys the the restored params:\n%s',
               restored_params_dict.keys() - params_dict.keys())

  splitkeys = {tuple(k.split('/')): v for k, v in params_dict.items()}
  params = flax.traverse_util.unflatten_dict(splitkeys)
  logging.info('Parameter summary after initialising from train state:')
  debug_utils.log_param_shapes(params)
  if hasattr(train_state, 'optimizer'):
    # TODO(dehghani): Remove support for flax optim.
    return train_state.replace(
        optimizer=train_state.optimizer.replace(
            target=flax.core.freeze(params)))
  else:
    return train_state.replace(params=flax.core.freeze(params))

