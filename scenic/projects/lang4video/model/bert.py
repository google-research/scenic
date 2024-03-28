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

"""BERT text encoder."""

import dataclasses
from typing import Any
from typing import Optional
from typing import Union

import flax
from flax.core import frozen_dict
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.lang4video.model.base_encoders import TextEncoder
import tensorflow as tf

from flaxformer.architectures.bert import bert
from flaxformer.architectures.bert import bert_checkpoint_converter
import flaxformer.architectures.bert.configs as bert_configs
import flaxformer.architectures.bert.heads as bert_heads


# Add the checkpoints:
CHECKPOINT = {}

CONFIGS: dict[str, bert_configs.BertConfig] = {
    'debug':
        bert_configs.BertBaseConfig(
            hidden_size=2,
            intermediate_dim=2,
            num_hidden_layers=1,
            num_attention_heads=2),
    'base':
        bert_configs.BertBaseConfig(),
    'large':
        bert_configs.BertLargeConfig(),
}


def convert_bert_mlm_head_params(tf_params: dict[str, tf.Tensor],
                                 prefix: str) -> dict[str, Any]:
  return {
      'dense':
          bert_checkpoint_converter.convert_dense_layer_params(
              tf_params, f'{prefix}transform/dense/'),
      'layer_norm':
          bert_checkpoint_converter.convert_layer_norm_params(
              tf_params, f'{prefix}transform/LayerNorm/'),
      **bert_checkpoint_converter.param_conversion_util.convert_tf_params(
          tf_params, {('bias',): 'output_bias'}, prefix),
  }


def load_params_from_tf_checkpoint(
    checkpoint_path: str
) -> tuple[frozen_dict.FrozenDict, frozen_dict.FrozenDict,
           frozen_dict.FrozenDict, frozen_dict.FrozenDict]:
  """Like the one with the same name plus returning the pretraining params."""
  ckpt_reader = tf.train.load_checkpoint(checkpoint_path)
  tf_params = {
      tf_name: ckpt_reader.get_tensor(tf_name)
      for tf_name in ckpt_reader.get_variable_to_dtype_map()
  }
  encoder_params = bert_checkpoint_converter.convert_bert_encoder_params(
      tf_params, 'bert/')
  pooler_params = bert_checkpoint_converter.param_conversion_util.convert_tf_params(
      tf_params, {
          ('dense', 'bias'): 'bias',
          ('dense', 'kernel'): 'kernel'
      }, 'bert/pooler/dense/')
  mlm_head_params = convert_bert_mlm_head_params(tf_params, 'cls/predictions/')
  nsp_head_params = bert_checkpoint_converter.param_conversion_util.convert_tf_params(
      tf_params, {
          (
              'mlp',
              'bias',
          ): 'output_bias',
          (
              'mlp',
              'kernel',
          ): 'output_weights',
      }, 'cls/seq_relationship/')
  return (frozen_dict.freeze(encoder_params), frozen_dict.freeze(pooler_params),
          frozen_dict.freeze(mlm_head_params),
          frozen_dict.freeze(nsp_head_params))


class BertTextEncoder(TextEncoder):
  """BERT text encoder."""

  config_name: str
  dtype: jnp.dtype = jnp.float32
  return_all_tokens: bool = False

  pretraining_mode: bool = False
  embedding_size: int = 512
  compute_mlm: bool = True

  @nn.compact
  def __call__(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      text: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      *,
      text_for_mlm: Optional[jnp.ndarray] = None,
      segment_ids_for_mlm: Optional[jnp.ndarray] = None,
      mask_for_mlm: Optional[jnp.ndarray] = None,
      masked_lm_positions: Optional[jnp.ndarray] = None,
      train: bool = False,
      debug: bool = False,
  ) -> Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    config = CONFIGS[self.config_name]
    config.dtype = self.dtype
    config.dropout_rate = 0.0

    # We leave the default eps because the LayerNorm is always computed in FP32.
    encoder = bert.BertEncoder(**dataclasses.asdict(config))
    mlm_head = bert_heads.MLMHead(
        encoder=encoder,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        kernel_init=config.kernel_init,
        dropout_rate=config.dropout_rate,
        dtype=config.dtype)
    pooler = bert_heads.BertPooler(
        kernel_init=config.kernel_init, dtype=config.dtype)
    cls_head = bert_heads.ClassifierHead(
        pooler=pooler,
        num_classes=self.embedding_size,
        kernel_init=config.kernel_init,
        dropout_rate=config.dropout_rate,
        enable_dropout=not train,
        dtype=config.dtype)

    encoded_tokens = encoder(
        token_ids=text,
        segment_ids=jnp.zeros_like(text),
        position_ids=jnp.arange(text.shape[-1]),
        input_mask=mask,
        enable_dropout=train)

    if self.pretraining_mode:
      encoded_sequence = cls_head(encoded_tokens)
      if self.compute_mlm:
        encoded_tokens_for_mlm = encoder(
            token_ids=text_for_mlm,
            segment_ids=segment_ids_for_mlm,
            position_ids=jnp.arange(text_for_mlm.shape[-1]),  # pytype: disable=attribute-error
            input_mask=mask_for_mlm,
            enable_dropout=train)
        mlm_logits = mlm_head(
            encoded_tokens_for_mlm, masked_positions=masked_lm_positions)
        return encoded_sequence, mlm_logits
      else:
        return encoded_sequence
    else:
      return encoded_tokens if self.return_all_tokens else encoded_tokens[:, 0]

  def get_pretrained_vars(
      self) -> tuple[frozen_dict.FrozenDict, frozen_dict.FrozenDict]:
    if self.config_name == 'debug':
      return flax.core.freeze({}), flax.core.freeze({})
    else:
      encoder_params, pooler_params, mlm_head_params, _ = load_params_from_tf_checkpoint(
          checkpoint_path=CHECKPOINTS[f'{self.config_name}-uncased'])
      params = {
          'BertEncoder_0': encoder_params,
      }
      if self.pretraining_mode:
        params['BertPooler_0'] = pooler_params
        if self.compute_mlm:
          params['MLMHead_0'] = mlm_head_params
      params = flax.core.freeze(params)
      params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
      return params, flax.core.freeze({})
