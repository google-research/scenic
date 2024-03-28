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

"""Implementation of Layout Denoise model."""

# pylint: disable=not-callable

from typing import Any, Dict, List, Tuple, Optional

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.projects.baselines import resnet
from scenic.projects.layout_denoise import base_model
from scenic.projects.layout_denoise.layers import common
from scenic.projects.layout_denoise.layers import embedding
from scenic.projects.layout_denoise.layers import predictor
from scenic.projects.layout_denoise.layers import transformer


class DeTRTransformer(nn.Module):
  """Layout Denoise DETR Transformer.

  Attributes:
    num_queries: Number of object queries. query_emb_size; Size of the embedding
      learned for object queries.
    query_emb_size: Size of the embedding learned for object queries.
    num_heads: Number of heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    return_intermediate_dec: If return the outputs from intermediate layers of
      the decoder.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_queries: int = 100
  query_emb_size: Optional[int] = None
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  return_intermediate_dec: bool = False
  normalize_before: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               image_inputs: jnp.ndarray,
               vh_inputs: jnp.ndarray,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies DeTRTransformer on the inputs.

    Args:
      image_inputs: Input data.
      vh_inputs: vh inputs.
      padding_mask: Binary mask containing 0 for padding tokens.
      pos_embedding: Positional Embedding to be added to the inputs.
      query_pos_emb: Positional Embedding to be added to the queries.
      train: Whether it is training.

    Returns:
      Output of the LayoutDETR transformer and output of the encoder.
    """
    encoder_norm = nn.LayerNorm() if self.normalize_before else None
    encoded = transformer.Encoder(
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        norm=encoder_norm,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='encoder')(
            image_inputs,
            padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            train=train)

    decoder_norm = nn.LayerNorm()
    output = transformer.Decoder(
        num_heads=self.num_heads,
        num_layers=self.num_decoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        return_intermediate=self.return_intermediate_dec,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        norm=decoder_norm,
        dtype=self.dtype,
        name='decoder')(
            vh_inputs,
            encoded,
            key_padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            query_pos_emb=query_pos_emb,
            train=train)
    return output  # pytype: disable=bad-return-type  # jax-ndarray


class VHOnlyModel(nn.Module):
  """Layout Denoise VH-only Transformer.

  Attributes:
    num_heads: Number of heads.
    num_encoder_layers: Number of encoder layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    return_intermediate_dec: If return the outputs from intermediate layers of
      the decoder.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """
  num_heads: int = 8
  num_encoder_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  normalize_before: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               vh_inputs: jnp.ndarray,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies DeTRTransformer on the inputs.

    Args:
      vh_inputs: vh inputs.
      padding_mask: Binary mask containing 0 for padding tokens.
      pos_embedding: Positional Embedding to be added to the inputs.
      train: Whether it is training.

    Returns:
      Output of the encoding of view hierarchy nodes.
    """
    encoder_norm = nn.LayerNorm() if self.normalize_before else None
    encoded = transformer.Encoder(
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        norm=encoder_norm,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='encoder')(
            vh_inputs,
            padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            train=train)

    return encoded  # pytype: disable=bad-return-type  # jax-ndarray


class MLPModel(nn.Module):
  """Layout Denoise MLP model for a single node.

  Attributes:
    num_encoder_layers: Number of encoder layers.
    qkv_dim: Dimension of the query/key/value.
    dropout_rate: Dropout rate.
    dtype: Data type of the computation (default: float32).
  """
  num_encoder_layers: int = 6
  hidden_dim: int = 512
  dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               vh_inputs: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies DeTRTransformer on the inputs.

    Args:
      vh_inputs: vh inputs.
      pos_embedding: Positional Embedding to be added to the inputs.
      padding_mask: Binary mask containing 0 for padding tokens.
      train: Whether it is training.

    Returns:
      Output of the MLP layers.
    """
    x = vh_inputs + pos_embedding
    for _ in range(self.num_encoder_layers):
      x = common.dense(x, self.hidden_dim, jnp.float32)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    return x  # pytype: disable=bad-return-type  # jax-ndarray


class LayoutDenoiseModel(nn.Module):
  """Layout denoise model.

  Attributes:
    modal_ranges: Modal ranges.
    num_classes: Number of classes.
    num_classes: Number of object classes.
    vocab_size: Vocabulary size.
    hidden_dim: Hidden dimension of the inputs to the model.
    num_queries: Number of object queries, ie detection slot. This is the
      maximal number of objects LayoutDETR can detect in a single image. For
      COCO, DETR paper recommends 100 queries.
    query_emb_size: Size of the embedding learned for object queries.
    transformer_num_heads: Number of transformer heads.
    transformer_num_encoder_layers: Number of transformer encoder layers.
    transformer_num_decoder_layers: Number of transformer decoder layers.
    transformer_qkv_dim: Dimension of the transformer query/key/value.
    transformer_mlp_dim: Dimension of the mlp on top of attention block.
    transformer_normalize_before: If use LayerNorm before attention/mlp blocks.
    backbone_num_filters: Num filters in the ResNet backbone.
    backbone_num_layers: Num layers in the ResNet backbone.
    aux_loss: If train with auxiliary loss.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Attention dropout rate.
    dtype: Data type of the computation (default: float32).
  """
  modal_ranges: List[int]
  num_classes: int
  vocab_size: int
  hidden_dim: int = 512
  query_emb_size: Optional[int] = None
  transformer_num_heads: int = 8
  transformer_num_encoder_layers: int = 6
  transformer_num_decoder_layers: int = 6
  transformer_qkv_dim: int = 512
  transformer_mlp_dim: int = 2048
  transformer_normalize_before: bool = False
  backbone_num_filters: int = 64
  backbone_num_layers: int = 50
  aux_loss: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  class_dropout_rate: float = 0.0
  model_type: str = 'full'
  pos_pattern: str = '1/4'

  @nn.compact
  def __call__(
      self,
      image: jnp.ndarray,
      obj_mask: jnp.ndarray,
      desc_id: jnp.ndarray,
      resource_id: jnp.ndarray,
      name_id: jnp.ndarray,
      boxes: jnp.ndarray,
      train: bool,
      *,
      task: str,
      padding_mask: Optional[jnp.ndarray] = None,
      update_batch_stats: bool = False,
      debug: bool = False) -> Dict[str, Any]:
    """Applies LayoutDETR model on the input.

    Args:
      image: Image data.
      obj_mask: A binary mask where valid is 1 and padding is 0.
      desc_id: description token ids.
      resource_id: resource-id token ids.
      name_id: android class name token ids.
      boxes: Object boxes.
      train:  Whether it is training.
      task: task name.
      padding_mask: Binary matrix with 0 at padded image regions.
      update_batch_stats: Whether update the batch statistics for the BatchNorms
        in the backbone. if None, the value of `train` flag will be used, i.e.
        we update the batch stat if we are in the train mode.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      Output: dit; that has 'pred_logits' and 'pred_boxes', and potentially
      'aux_outputs'.
    """
    assert self.hidden_dim == self.transformer_qkv_dim

    num_queries = self.modal_ranges[1] - self.modal_ranges[0]
    token_embder = embedding.TokenEmbedding(
        hidden_dim=self.hidden_dim, vocab_size=self.vocab_size)

    structure_dict = embedding.StructureEmbedding(
        hidden_dim=self.hidden_dim,
        num_queries=num_queries,
        dropout_rate=self.dropout_rate)(
            obj_mask,
            desc_id,
            resource_id,
            name_id,
            boxes,
            # clickable,
            task=task,
            token_embder=token_embder,
            pos_pattern=self.pos_pattern,
            train=train)

    if self.model_type == 'full':
      # Full model using both image and view hierarchy structure.
      cnn = resnet.ResNet(
          num_outputs=None,
          num_filters=self.backbone_num_filters,
          num_layers=self.backbone_num_layers,
          dtype=self.dtype,
          name='backbone')
      image_dict = embedding.ImageEmbedding(
          hidden_dim=self.hidden_dim,
          backbone_num_filters=self.backbone_num_filters,
          backbone_num_layers=self.backbone_num_layers,
          name='image_embedding')(
              cnn=cnn,
              images=image,
              train=train,
              padding_mask=padding_mask,
              update_batch_stats=update_batch_stats)
      layout_detr = DeTRTransformer(
          num_queries=num_queries,
          query_emb_size=self.query_emb_size,
          num_heads=self.transformer_num_heads,
          num_encoder_layers=self.transformer_num_encoder_layers,
          num_decoder_layers=self.transformer_num_decoder_layers,
          qkv_dim=self.transformer_qkv_dim,
          mlp_dim=self.transformer_mlp_dim,
          normalize_before=self.transformer_normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          return_intermediate_dec=self.aux_loss)
      decoder_output = layout_detr(
          image_dict['content_emb'],
          structure_dict['content_emb'],
          padding_mask=image_dict['mask'],
          pos_embedding=image_dict['pos_emb'],
          query_pos_emb=structure_dict['pos_emb'],
          train=train)
    elif self.model_type == 'vh_only':
      # Model only using view hierarchy structure.
      layout_vh_only = VHOnlyModel(
          num_heads=self.transformer_num_heads,
          num_encoder_layers=self.transformer_num_encoder_layers,
          qkv_dim=self.transformer_qkv_dim,
          mlp_dim=self.transformer_mlp_dim,
          normalize_before=self.transformer_normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate)
      decoder_output = layout_vh_only(
          vh_inputs=structure_dict['content_emb'],
          pos_embedding=structure_dict['pos_emb'],
          padding_mask=structure_dict['mask'],
          train=train)
    elif self.model_type == 'mlp':
      # Model only using individual nodes in view hiearchy without structure.
      layout_mlp = MLPModel(
          num_encoder_layers=self.transformer_num_encoder_layers,
          hidden_dim=self.hidden_dim,
          dropout_rate=self.dropout_rate)
      decoder_output = layout_mlp(
          vh_inputs=structure_dict['content_emb'],
          pos_embedding=structure_dict['pos_emb'],
          padding_mask=structure_dict['mask'],
          train=train)

    def output_projection(params):
      model_output = params['multimodal_outputs']
      return output_projection_pamp(model_output, train)

    def output_projection_pamp(model_output, train):
      pred_logits = predictor.ObjectClassPredictor(
          num_classes=self.num_classes, dropout_rate=self.class_dropout_rate)(
              model_output, deterministic=not train)
      return {
          'pred_logits': pred_logits,
      }

    params = {
        'multimodal_outputs': decoder_output,
        'train': train,
    }

    return common.create_output(
        output_projection,
        params=params,
        aux_loss=self.aux_loss,
        layout_model_pamp=output_projection_pamp)


class LayoutModel(base_model.LayoutDenoiseBaseModel):
  """Layout model."""

  def build_flax_model(self):
    return LayoutDenoiseModel(
        modal_ranges=self.config['modal_ranges'],
        num_classes=self.config['num_classes'],
        vocab_size=self.config['vocab_size'],
        hidden_dim=self.config.get('hidden_dim', 512),
        query_emb_size=self.config.get('query_emb_size', None),
        transformer_num_heads=self.config.get('transformer_num_heads', 8),
        transformer_num_encoder_layers=self.config.get(
            'transformer_num_encoder_layers', 6),
        transformer_num_decoder_layers=self.config.get(
            'transformer_num_decoder_layers', 6),
        transformer_qkv_dim=self.config.get('transformer_qkv_dim', 512),
        transformer_mlp_dim=self.config.get('transformer_mlp_dim', 2048),
        transformer_normalize_before=self.config.get(
            'transformer_normalize_before', False),
        backbone_num_filters=self.config.get('backbone_num_filters', 64),
        backbone_num_layers=self.config.get('backbone_num_layers', 50),
        aux_loss=self.config.get('aux_loss', False),
        dropout_rate=self.config.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.get('attention_dropout_rate', 0.0),
        dtype=jnp.float32,
        class_dropout_rate=self.config.get('class_dropout_rate', 0.0),
        model_type=self.config.get('model_type', 'full'),
        pos_pattern=self.config.get('pos_pattern', '1/4'))

  def default_flax_model_config(self):
    return ml_collections.ConfigDict(
        dict(
            modal_ranges=[42 * 42, 42 * 42 + 101],
            num_classes=30,
            vocab_size=30_000,
            hidden_dim=32,
            query_emb_size=None,
            transformer_num_heads=2,
            transformer_num_encoder_layers=1,
            transformer_num_decoder_layers=1,
            transformer_qkv_dim=32,
            transformer_mlp_dim=32,
            transformer_normalize_before=False,
            backbone_num_filters=32,
            backbone_num_layers=1,
            aux_loss=False,
            panoptic=False,
            dropout_rate=0.0,
            attention_dropout_rate=0.0))
