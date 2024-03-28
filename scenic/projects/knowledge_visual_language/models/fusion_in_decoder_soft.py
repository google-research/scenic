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

"""Baseline for Image to Text Models.

B = batch size
H = height
W = width
N = number of image tokens
I = Input sequence length
O = Ouput sequence length
d = hidden dims
C = number of vocabulary
K = number of candidate
L = sequence length of retrieved document
M = sequence length of compressed tokens
"""
from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.projects.knowledge_visual_language.models import constants
from scenic.projects.knowledge_visual_language.models import layers
from scenic.projects.knowledge_visual_language.models import losses
from scenic.projects.knowledge_visual_language.models import metrics
from scenic.projects.knowledge_visual_language.models import vit as vit_model
from scenic.projects.t5 import layers as t5_model
from scenic.projects.t5 import model as t5_pretrained


class VisionLanguageModule(nn.Module):
  """Basic ViT + T5 vision language model."""

  config: ml_collections.ConfigDict

  def setup(self):
    t5_config = t5_pretrained.CONFIGS[self.config.t5_name]
    self.t5_config = t5_config
    t5_config['dropout_rate'] = self.config.dropout_rate
    self.ndim = t5_config['emb_dim']
    self.dropout_rate = t5_config['dropout_rate']
    self.key_dim = self.config.key_dim
    self.dtype = t5_config['dtype']
    # Shared token embedding for T5 encoder & Decoder
    self.shared_token_embedder = t5_model.t5_layers.Embed(
        num_embeddings=t5_config['vocab_size'],
        features=self.ndim,
        dtype=self.dtype,
        attend_dtype=self.dtype,  # For logit training stability.
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='shared_token_embedder',
    )
    # Pre-Trained Lower T5 Decoder
    self.out_decoder = t5_model.T5Decoder(
        **t5_config,
        shared_embedding=self.shared_token_embedder,
        name='out_decoder'
    )
    # Uni-Modal Text Encoding (Pre-Trained Lower T5 Encoder)
    self.text_encoder = layers.LowerT5Encoder(
        **t5_config,
        num_fusion_layers=self.config.num_fusion_layers,
        shared_embedding=self.shared_token_embedder,
        name='text_encoder'
    )
    # Multi-Modal Fusion Encoder (Pre-Trained Upper T5 Encoder)
    self.fusion_encoder = layers.FusedT5Encoder(
        **t5_config,
        num_fusion_layers=self.config.num_fusion_layers,
        name='fusion_encoder'
    )
    # Visual Encoding (Pre-Trained ViT)
    self.img_encoder = vit_model.Model(
        num_classes=self.ndim,
        dropout=self.dropout_rate,
        name='img_encoder',
        variant=self.config.vit_name,
        head_zeroinit=False,
        dtype=jnp.bfloat16,
        num_frozen_layers=self.config.get('vit_num_frozen_layers', -1),
        pool_type='gap',
    )
    self.dropout = nn.Dropout(rate=0.2)

  def get_base_encoded(
      self,
      image=None,
      text_tokens=None,
      train=False,
      random_drop_image=False,
      bsz=None,
      frozen_base=True,
  ):
    if bsz is None:
      if text_tokens is not None:
        bsz = len(text_tokens)
      elif image is not None:
        bsz = len(image)
    if text_tokens is not None:
      text_query, text_mask = self.text_encoder(
          encoder_input_tokens=text_tokens,
          use_dropout=train,
          frozen_base=frozen_base,
      )  # B×I×d
    else:
      text_query = jnp.zeros([bsz, 1, self.ndim], dtype=self.dtype)
      text_mask = jnp.zeros([bsz, 1], dtype=self.dtype)
    if image is not None:
      img_query, img_emb = self.encode_image(image, train=train)
      n_img_tokens = img_query.shape[1]
    else:
      n_img_tokens = 1
      img_query = jnp.zeros([bsz, n_img_tokens, self.ndim], dtype=self.dtype)
      img_emb = jnp.zeros([bsz, self.ndim], dtype=self.dtype)
    if train and random_drop_image:
      image_mask = jax.random.bernoulli(
          self.make_rng('dropout'), p=1 - 0.2, shape=(bsz, 1)
      ).astype(self.dtype)
      img_emb = img_emb * image_mask
      image_mask = jnp.repeat(image_mask, repeats=n_img_tokens, axis=1)
    else:
      image_mask = jnp.ones([bsz, n_img_tokens], dtype=self.dtype)
    base_masks = jnp.concatenate([text_mask, image_mask], axis=1)
    return [text_query, img_query], base_masks, img_emb


class FusionInDecoderSoftModule(VisionLanguageModule):
  """Modification of FID (https://arxiv.org/pdf/2007.01282.pdf) model.

  Take continous embedding of retrieved document at middle fusion layer
  instead of whole sequence at input.
  """

  config: ml_collections.ConfigDict

  def setup(self):
    super().setup()
    self.n_compressed_tokens = self.config.n_compressed_tokens
    # Project retrieved knowledge into encoder space
    self.value_perceiver = layers.PerceiverEncoder(
        **self.t5_config,
        num_fusion_layers=self.config.num_fusion_layers,
        perceiver_output_dim=self.n_compressed_tokens,
        name='value_perceiver'
    )
    # Query & Key Head for Retrieval
    self.compress_head = nn.Dense(
        features=self.key_dim, dtype=self.dtype, name='head_out', use_bias=False
    )
    self.query_head = layers.TransformerHead(
        **self.t5_config,
        num_head_layers=self.config.num_fusion_layers,
        out_head=self.compress_head,
        key_dim=self.key_dim,
        name='query_head'
    )
    self.key_head = layers.TransformerHead(
        **self.t5_config,
        num_head_layers=self.config.num_fusion_layers,
        out_head=self.compress_head,
        key_dim=self.key_dim,
        name='key_head'
    )
    self.att_transform = layers.AffineTransform()

  def compress_and_pool_key(self, h, mask):
    window_size = self.n_stride
    pooled_tokens = nn.avg_pool(
        h[:, self.n_compressed_tokens :, :],
        window_shape=(window_size,),
        strides=(self.n_stride,),
    )
    pooled_tokens = jnp.concatenate(
        (h[:, : self.n_compressed_tokens, :], pooled_tokens), axis=1
    )
    pooled_mask = jnp.squeeze(
        -nn.max_pool(
            jnp.expand_dims(-mask[:, self.n_compressed_tokens :], axis=-1),
            window_shape=(window_size,),
            strides=(self.n_stride,),
        )
    )
    pooled_mask = jnp.concatenate(
        (mask[:, : self.n_compressed_tokens], pooled_mask), axis=1
    )
    # Total: 10 + 512 / n_stride = 42 tokens
    return pooled_tokens, pooled_mask

  def compress_key(self, h, mask):
    pooled_tokens = h[:, : self.n_compressed_tokens, :]
    pooled_mask = mask[:, : self.n_compressed_tokens]
    return pooled_tokens, pooled_mask

  def encode_knowledge(
      self,
      retr_texts,
      retr_images=None,
      bsz=None,
      train=False,
      random_drop_image=False,
      frozen_base=True,
  ):
    retr_tokens, retr_masks, retr_img_emb = self.get_base_encoded(
        bsz=bsz,
        image=retr_images,
        text_tokens=retr_texts,
        train=train,
        random_drop_image=random_drop_image,
        frozen_base=frozen_base,
    )
    retr_tokens = jnp.concatenate(retr_tokens, axis=1)  # B×(I+N)×d
    retr_keys = self.key_head(
        encoded_emb=retr_tokens, encoder_mask=retr_masks, use_dropout=train
    )  # B×(I+N)×d -> B×d
    compressed_val, compressed_mask, disentangle_reg = self.value_perceiver(
        encoded=retr_tokens, encoded_mask=retr_masks, use_dropout=train
    )

    return (
        retr_keys,
        compressed_val,
        compressed_mask,
        retr_img_emb,
        disentangle_reg,
    )

  def encode_query(
      self,
      encoder_input_image,
      encoder_input_tokens,
      train=False,
      frozen_base=True,
  ):
    bsz = encoder_input_image.shape[0]
    base_vals, base_masks, _ = self.get_base_encoded(
        bsz=bsz,
        image=encoder_input_image,
        text_tokens=encoder_input_tokens,
        train=train,
        frozen_base=frozen_base,
    )
    base_vals = self.dropout(
        jnp.concatenate(base_vals, axis=1), deterministic=not train
    )  # B×(I+N)×d
    base_query = self.query_head(
        encoded_emb=base_vals, encoder_mask=base_masks, use_dropout=train
    )
    return base_vals, base_masks, base_query

  def encode_topk_knowledge(
      self,
      bsz,
      retr_texts,
      retr_images=None,
      train=False,
      random_drop_image=False,
      frozen_base=True,
  ):
    k, l = retr_texts.shape[1], retr_texts.shape[2]
    retr_texts = jnp.reshape(retr_texts, (bsz * k, l))
    if retr_images is not None:
      image_shape = (bsz * k,) + retr_images.shape[2:]
      retr_images = jnp.reshape(retr_images, image_shape)
    (
        retr_keys,
        compressed_val,
        compressed_mask,
        retr_img_emb,
        disentangle_reg,
    ) = self.encode_knowledge(
        retr_texts,
        retr_images,
        bsz=bsz * k,
        train=train,
        random_drop_image=random_drop_image,
        frozen_base=frozen_base,
    )
    n_tokens = compressed_val.shape[1]
    retr_keys = jnp.reshape(retr_keys, (bsz, k, self.key_dim))
    compressed_val = jnp.reshape(
        compressed_val, (bsz, k, n_tokens, self.ndim)
    )  # B×K×M×d
    compressed_mask = jnp.reshape(compressed_mask, (bsz, k, n_tokens))
    return (
        retr_keys,
        compressed_val,
        compressed_mask,
        retr_img_emb,
        disentangle_reg,
    )

  def encode_image(self, image, train=False):
    _, out = self.img_encoder(image, train=train)  # B×W×H×3 -> B×N×d
    img_query = jnp.asarray(out['logits_2d'] * 4, self.dtype)
    n_img_tokens = img_query.shape[1] * img_query.shape[2]
    img_query = jnp.reshape(img_query, [-1, n_img_tokens, self.ndim])
    img_emb = jnp.asarray(out['head_input'], self.dtype)
    return img_query, img_emb

  def fuse_topk_knowledge(
      self,
      base_query,
      base_vals,
      base_masks,
      retr_keys,
      retr_vals,
      retr_masks,
      train=False,
  ):
    (bsz, k, n_tokens) = retr_vals.shape[:3]
    retr_vals = jnp.reshape(
        retr_vals, (bsz, k * n_tokens, self.ndim)
    )  # B×(M*K)×d
    retr_scores = jnp.einsum('bd,bkd->bk', base_query, retr_keys)
    retr_scores = jax.nn.softmax(self.att_transform(retr_scores), axis=-1) * k
    retr_masks = jnp.reshape(retr_masks, (bsz, k * n_tokens))
    att_mask = [
        jnp.ones([bsz, base_vals.shape[1]]),
        jnp.repeat(retr_scores, repeats=n_tokens, axis=-1),
    ]
    att_mask = jnp.expand_dims(jnp.concatenate(att_mask, axis=-1), axis=-1)
    fused_query, fused_mask, attn_weights_all_layers = self.fusion_encoder(
        encoder_input_embs=base_vals,
        fused_input_embs=retr_vals,
        encoder_mask=base_masks,
        fused_mask=retr_masks,
        att_mask=att_mask,
        use_dropout=train,
        output=True,
    )  # B×(I+N+M*K)×d
    return fused_query, fused_mask, retr_scores, attn_weights_all_layers

  def __call__(
      self,
      decoder_input_tokens,  # B×O
      decoder_target_tokens,  # B×O
      encoder_input_image=None,  # B×W×H×3
      encoder_input_tokens=None,  # B×I
      retr_texts=None,  # B×K×L
      retr_images=None,  # B×K×W×H×3
      train=False,
      decode=False,
      fuse_retrieval=True,
      max_decode_length=None,
      debug: bool = False,
      in_batch_neg: bool = False,
      frozen_base=True,
      **args
  ):
    """Conduct supervised retrieval-augmented training with given retrieved documents.

    Args:
      decoder_input_tokens:  # B×O.
      decoder_target_tokens:  # B×O.
      encoder_input_image:  # B×W×H×3.
      encoder_input_tokens: # B×I.
      retr_texts:  # B×K×L.
      retr_images:  # B×K×W×H×3.
      train: whether using train mode.
      decode: whether in decode mode.
      fuse_retrieval: whether use input retrieval docs.
      max_decode_length: maximum decode token length.
      debug: whether use debug mode.
      in_batch_neg: whether use in-batch contastive learning.
      frozen_base: whether froze the whole encoder.
      **args: other possible arguments.

    Returns:
      output dictionary containing final and intermediate results.
    """
    bsz = decoder_input_tokens.shape[0]
    base_vals, base_masks, query_img_emb = self.get_base_encoded(
        bsz=bsz,
        image=encoder_input_image,
        text_tokens=encoder_input_tokens,
        train=train,
        frozen_base=frozen_base,
    )  # B×N×d, B×I×d
    out_dict = {
        'query_img_emb': query_img_emb,
        'text_query': base_vals[0],
        'image_query': base_vals[1],
    }
    base_vals = jnp.concatenate(base_vals, axis=1)  # B×(I+N)×d
    if retr_texts is not None:
      retr_keys, retr_vals, retr_masks, retr_img_emb, disentangle_reg = (
          self.encode_topk_knowledge(
              bsz=bsz,
              retr_images=retr_images,
              retr_texts=retr_texts,
              train=train,
              random_drop_image=True,
          )
      )
      base_query = self.query_head(
          encoded_emb=base_vals, encoder_mask=base_masks, use_dropout=train
      )  # B×(I+N)×d -> B×d
      out_dict['disentangle_reg'] = disentangle_reg
      out_dict['retr_img_emb'] = retr_img_emb
      out_dict['base_query'] = base_query
      out_dict['retr_keys'] = retr_keys
      out_dict['retr_vals'] = retr_vals

    if fuse_retrieval and retr_texts is not None:
      # fuse top-k retrieved knowledge (or no fusion)
      if in_batch_neg and retr_vals.shape[1] == 1:
        # retr_vals: B×1×M×d -> B×2×M×d, retr_keys: B×1×d -> B×2×M×d
        retr_vals = jnp.concatenate(
            (retr_vals, jnp.roll(retr_vals, shift=1, axis=0)), axis=1
        )
        retr_keys = jnp.concatenate(
            (retr_keys, jnp.roll(retr_keys, shift=1, axis=0)), axis=1
        )
        retr_masks = jnp.concatenate(
            (retr_masks, jnp.roll(retr_masks, shift=1, axis=0)), axis=1
        )

      fused_emb, fused_mask, retr_scores, attn_weights_all_layers = (
          self.fuse_topk_knowledge(
              base_query=base_query,
              base_vals=base_vals,
              base_masks=base_masks,
              retr_keys=retr_keys,
              retr_vals=retr_vals,
              retr_masks=retr_masks,
              train=train,
          )
      )  # B×(I+N+M*K)×d
      out_dict['retr_scores'] = retr_scores
    else:
      # only fuse input image and text
      fused_emb, fused_mask, attn_weights_all_layers = self.fusion_encoder(
          fused_input_embs=base_vals, fused_mask=base_masks, use_dropout=train
      )  # B×(I+N)×d
    # generate decoding results.
    out_dict['attn_weights_all_layers'] = attn_weights_all_layers
    out_dict['predicted_logits'] = self.out_decoder(
        encoded=fused_emb,
        decoder_input_tokens=decoder_input_tokens,
        encoder_input_tokens=fused_mask,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=train,
        decode=decode,
        max_decode_length=max_decode_length,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
    )
    return out_dict


class FIDSoftModel(base_model.BaseModel):
  """FID model."""

  def build_flax_model(self) -> nn.Module:
    return FusionInDecoderSoftModule(self.config.model)

  def loss_function_dict(
      self, output: constants.JTensorDict, batch: constants.JTensorDict
  ) -> Dict[str, Any]:
    """Returns negative loglikelihood (NLL) of the target sentence.

    Args:
      output: Output of model in OrderedDict.
      batch: Batch of data that has 'decoder_target' as ground-truth.

    Returns:
      Total loss.
    """
    gen_loss = losses.nll_loss(
        targets=batch['decoder_target_tokens'],
        pred=output['predicted_logits'],
        target_masks=batch['decoder_target_tokens'] > 0,
        label_smoothing=self.config.model.get('label_smoothing'),
    )
    loss_dict = {'gen_loss': gen_loss}
    if output['supervised_retrieval']:
      retr_loss, (retr_acc, s0, s1) = losses.contrastive_loss(
          query_emb=output['base_query'],
          key_emb=output['retr_keys'],
          temperature=self.config.model.get('temperature'),
      )
      loss_dict['retr_loss'] = retr_loss
      loss_dict['retr_acc'] = retr_acc
      loss_dict['s0'] = s0
      loss_dict['s1'] = s1
    else:
      loss_dict['retr_loss'] = -1
      loss_dict['retr_acc'] = -1
      loss_dict['s0'] = -1
      loss_dict['s1'] = -1
    return loss_dict

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(outputs,
      batch)```
    """

    return metrics.token_accuracy
