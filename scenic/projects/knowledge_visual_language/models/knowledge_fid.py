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
import functools
from typing import Any, Dict, Mapping, Optional, Tuple, List

from absl import logging
import flax.linen as nn
import jax
from jax.experimental import host_callback  # pylint: disable=unused-import
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.knowledge_visual_language.data import data_utils
from scenic.projects.knowledge_visual_language.models import constants
from scenic.projects.knowledge_visual_language.models import fusion_in_decoder_soft
from scenic.projects.knowledge_visual_language.models import layers
from scenic.projects.knowledge_visual_language.models import local_memory
from scenic.projects.knowledge_visual_language.models import losses
from scenic.projects.knowledge_visual_language.models import metrics
from t5x import decoding

local_kb = local_memory.kb


class KnowledgeFIDModule(fusion_in_decoder_soft.FusionInDecoderSoftModule):
  """FID model (https://arxiv.org/pdf/2007.01282.pdf) with a retrieval module over a knowledge memory."""

  retr_k: int
  data_k: int
  axis_index_groups: Optional[List[List[int]]] = None
  across_index_groups: Optional[List[List[int]]] = None

  def setup(self):
    super().setup()
    self.local_keys = self.variable(
        'memory',
        'keys',
        functools.partial(jnp.zeros, dtype=jnp.bfloat16),
        (local_kb.n_data_per_shard, self.key_dim),
    )
    self.local_dataset_idxs = self.variable(
        'memory',
        'idxs',
        functools.partial(jnp.zeros, dtype=jnp.int16),
        (local_kb.n_data_per_shard * local_kb.n_local_device),
    )
    self.dataset_gate = nn.Dense(
        features=local_kb.n_kb_dataset, dtype=self.dtype, name='dataset_gate'
    )

  def _get_corpus_scores(self, corpus_scores, topk_ids):
    corpus_ids = jnp.take(self.local_dataset_idxs.value, topk_ids, axis=0)
    return layers.batch_index_select(corpus_scores, corpus_ids), corpus_ids

  def _dist_mips_local(
      self,
      query,
      corpus_scores,
      local_device_id,
      recall_target=0.99,
      exact=False,
  ):
    logging.info('mips local!!!')
    logging.info(self.local_keys.value.shape)
    logging.info(local_kb.n_data)
    host_query = jax.lax.all_gather(
        x=query,
        axis_name='batch',
        axis=0,
        axis_index_groups=self.axis_index_groups,
        tiled=True,
    )
    logging.info(host_query.shape)

    local_scores = jax.lax.dot(host_query, self.local_keys.value.transpose())
    if exact:
      local_topk_scores, local_topk_ids = jax.lax.top_k(
          local_scores, k=self.retr_k
      )
    else:
      local_topk_scores, local_topk_ids = jax.lax.approx_max_k(
          local_scores,
          k=self.retr_k,
          recall_target=recall_target,
          reduction_input_size_override=local_kb.n_data,
          aggregate_to_topk=True,
      )
    local_topk_ids_offset = (
        local_topk_ids + local_device_id * local_kb.n_data_per_shard
    )
    # local_topk_scores: bsz * k
    logging.info(local_topk_ids.shape)
    host_topk_scores = jax.lax.all_to_all(
        x=local_topk_scores,
        axis_name='batch',
        split_axis=0,
        concat_axis=1,
        axis_index_groups=self.axis_index_groups,
        tiled=True,
    )
    host_topk_ids = jax.lax.all_to_all(
        x=local_topk_ids_offset,
        axis_name='batch',
        split_axis=0,
        concat_axis=1,
        axis_index_groups=self.axis_index_groups,
        tiled=True,
    )
    # host_topk_scores: per_bsz * (n_device * k)
    logging.info(host_topk_scores.shape)

    host_corpus_scores, host_corpus_ids = self._get_corpus_scores(
        corpus_scores, host_topk_ids
    )
    host_topk_scores, host_rank_ids = jax.lax.top_k(
        host_topk_scores * host_corpus_scores, k=self.retr_k
    )
    # host_topk_scores: per_bsz * retr_k

    host_data_ids = layers.batch_index_select(
        host_topk_ids, host_rank_ids[:, : self.data_k]
    )
    host_memory_ids = layers.batch_index_select(
        host_topk_ids, host_rank_ids[:, self.data_k :]
    )
    host_corpus_ids = layers.batch_index_select(host_corpus_ids, host_rank_ids)

    logging.info('data and memory shape!!!')
    logging.info(host_data_ids.shape)
    logging.info(host_memory_ids.shape)

    # retrieve memory
    args = (host_data_ids, host_memory_ids)

    ret_memory, ret_data = host_callback.call(
        local_memory.local_retrieve_memory,
        args,
        result_shape=local_kb.local_ret_specs,
    )

    ret_memory['masks'] = jnp.ones(ret_memory['values'].shape[:3]).astype(bool)

    for k in ret_memory:
      logging.info(k)
      logging.info(ret_memory[k].shape)
      logging.info(ret_memory[k].dtype)

    return (
        host_topk_scores,
        ret_memory,
        ret_data,
        host_topk_ids,
        host_rank_ids,
        host_corpus_ids,
    )

  def _dist_mips_across(
      self,
      query,
      corpus_scores,
      local_device_id,
      recall_target=0.99,
      exact=False,
  ):
    # must have n_host > retr_k
    logging.info('mips global!!!')
    logging.info(self.local_keys.value.shape)
    logging.info(local_kb.n_data)
    n_local_device = len(self.across_index_groups)
    logging.info(n_local_device)
    global_query = jax.lax.all_gather(
        x=query, axis_name='batch', axis=0, tiled=True
    )
    logging.info(global_query.shape)
    # global_query: (per_bsz * n_local_device * n_hosts) * d
    global_corpus_scores = jax.lax.all_gather(
        x=corpus_scores, axis_name='batch', axis=0, tiled=True
    )
    logging.info(global_corpus_scores.shape)
    # global_corpus_scores: (per_bsz * n_local_device * n_hosts) * n_kb

    local_scores = jax.lax.dot(global_query, self.local_keys.value.transpose())
    local_k = local_kb.k
    if exact:
      local_topk_scores, local_topk_ids = jax.lax.top_k(local_scores, k=local_k)
    else:
      local_topk_scores, local_topk_ids = jax.lax.approx_max_k(
          local_scores,
          k=local_k,
          recall_target=recall_target,
          reduction_input_size_override=local_kb.n_data,
          aggregate_to_topk=True,
      )

    local_topk_ids_offset = (
        local_topk_ids + local_device_id * local_kb.n_data_per_shard
    )
    logging.info(local_topk_ids.shape)
    # local_topk_ids: (per_bsz * n_local_device * n_hosts) * K
    host_topk_scores = jax.lax.all_gather(
        x=local_topk_scores,
        axis_name='batch',
        axis=1,
        axis_index_groups=self.axis_index_groups,
        tiled=True,
    )
    logging.info(host_topk_scores.shape)
    # host_topk_scores: (per_bsz * n_local_device * n_hosts) * (n_hosts * K)
    host_topk_ids = jax.lax.all_gather(
        x=local_topk_ids_offset,
        axis_name='batch',
        axis=1,
        axis_index_groups=self.axis_index_groups,
        tiled=True,
    )
    # host_topk_ids: (per_bsz * n_local_device * n_hosts) * (n_hosts * K)

    host_corpus_scores, host_corpus_ids = self._get_corpus_scores(
        global_corpus_scores, host_topk_ids
    )
    # host_corpus_scores: (per_bsz * n_local_device * n_hosts) * (n_hosts * K)
    host_topk_scores, host_rank_ids = jax.lax.top_k(
        host_topk_scores * host_corpus_scores, k=local_k
    )
    # host_topk_scores: (per_bsz * n_local_device * n_hosts) * K
    host_topk_ids = layers.batch_index_select(host_topk_ids, host_rank_ids)
    logging.info(host_topk_ids.shape)
    # host_topk_ids: (per_bsz * n_local_device * n_hosts) * K
    host_topk_ids = jnp.reshape(host_topk_ids, (-1, n_local_device, local_k))
    host_topk_ids = host_topk_ids[:, local_device_id]
    logging.info(host_topk_ids.shape)
    # host_topk_ids: (per_bsz * n_hosts)
    host_topk_scores = jnp.reshape(
        host_topk_scores, (-1, n_local_device, local_k)
    )
    host_topk_scores = host_topk_scores[:, local_device_id]
    logging.info('host_topk_scores')
    logging.info(host_topk_scores.shape)

    ret_memory, ret_data = host_callback.call(
        local_memory.retrieve_top_memory,
        (host_topk_ids),
        result_shape=local_kb.ret_top_specs,
    )

    global_topk_scores = jax.lax.all_to_all(
        x=host_topk_scores,
        axis_name='batch',
        split_axis=0,
        concat_axis=1,
        axis_index_groups=self.across_index_groups,
        tiled=True,
    )
    logging.info('global_topk_scores')
    logging.info(global_topk_scores.shape)
    # global_topk_scores: per_bsz * (n_device * k)
    global_topk_scores, global_rank_ids = jax.lax.top_k(
        global_topk_scores, k=self.retr_k
    )
    logging.info(global_topk_scores.shape)
    # global_topk_scores: per_bsz * retr_k
    global_data_ids = global_rank_ids[:, : int(self.data_k)]
    global_memory_ids = global_rank_ids[:, int(self.data_k) :]

    def _gather_val(local_ret_vals, top_ids):
      logging.info(local_ret_vals.shape)
      global_ret_vals = jax.lax.all_to_all(
          x=local_ret_vals,
          axis_name='batch',
          split_axis=0,
          concat_axis=1,
          axis_index_groups=self.across_index_groups,
          tiled=True,
      )
      logging.info(global_ret_vals.shape)
      # global_ret_vals: per_bsz * (n_device * k) * dshape
      global_ret_vals = layers.batch_index_select(global_ret_vals, top_ids)
      logging.info(global_ret_vals.shape)
      # global_ret_vals: per_bsz * retr_k * dshape
      return global_ret_vals

    logging.info('_gather_val!!!')

    ret_memory = jax.tree_util.tree_map(
        lambda local_val: _gather_val(local_val, global_memory_ids), ret_memory
    )

    ret_data = jax.tree_util.tree_map(
        lambda local_val: _gather_val(local_val, global_data_ids), ret_data
    )

    ret_memory['masks'] = jnp.ones(ret_memory['values'].shape[:3]).astype(bool)

    for k in ret_memory:
      logging.info(k)
      logging.info(ret_memory[k].shape)
      logging.info(ret_memory[k].dtype)

    host_corpus_ids = layers.batch_index_select(host_corpus_ids, host_rank_ids)
    host_corpus_ids = jnp.reshape(
        host_corpus_ids, (-1, n_local_device, local_k)
    )[:, local_device_id]
    logging.info('corpus_ids')
    logging.info(host_corpus_ids.shape)
    # host_corpus_ids: (per_bsz * n_local_device, k)

    global_corpus_ids = jax.lax.all_to_all(
        x=host_corpus_ids,
        axis_name='batch',
        split_axis=0,
        concat_axis=1,
        axis_index_groups=self.across_index_groups,
        tiled=True,
    )
    logging.info(global_corpus_ids.shape)
    # global_corpus_ids: (per_bsz, n_local_device)
    global_corpus_ids = layers.batch_index_select(
        global_corpus_ids, global_rank_ids
    )
    logging.info(global_corpus_ids.shape)
    # global_corpus_ids: (per_bsz, 10)
    return (
        global_topk_scores,
        ret_memory,
        ret_data,
        local_topk_ids,
        global_rank_ids,
        global_corpus_ids,
    )

  def t5_decode(
      self,
      encoded,
      encoder_input_tokens: jnp.ndarray,  # Only needed for masks.
      decoder_input_tokens: jnp.ndarray,
      decoder_target_tokens: jnp.ndarray,
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
  ):
    """wraps _t5_decoder call (no packing) to enable autoregressive decoding."""
    # Without this wrapper flax.model.apply does not know self._t5_decoder yet
    # when doing a single (autoregressive) decode step.
    return self.out_decoder(
        encoded=encoded,
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
    )

  def __call__(
      self,
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_input_image=None,
      encoder_input_tokens=None,
      retr_texts=None,
      retr_images=None,
      device_id=0,
      train=False,
      decode=False,
      max_decode_length=None,
      use_memory=False,
      use_psudo_retr=False,
      retrieve_local=False,
      no_memory=False,
      debug=False,
      frozen_base=True,
      only_encode=False,
      **args
  ):
    """Conduct online retrieval and retrieval-augmented generataion.

    Args:
      decoder_input_tokens:  # B×O.
      decoder_target_tokens:  # B×O.
      encoder_input_image:  # B×W×H×3.
      encoder_input_tokens: # B×I.
      retr_texts:  # B×K×L.
      retr_images:  # B×K×W×H×3.
      device_id: index of TPU device.
      train: whether using train mode.
      decode: whether in decode mode.
      max_decode_length: maximum decode token length.
      use_memory: whether use on-device memory.
      use_psudo_retr: whether to use psudo retrieved groundtruth for guidance.
      retrieve_local: whether only retrieve in local host or across hosts.
      no_memory: whether not using any retrieval.
      debug: whether use debug mode.
      frozen_base: whether froze the whole encoder.
      only_encode: skip decoding and only return encoded tokens.
      **args: other possible arguments.

    Returns:
      output dictionary containing final and intermediate results.
    """
    bsz = decoder_input_tokens.shape[0]

    out_dict = {}
    base_vals, base_masks, base_query = self.encode_query(
        encoder_input_image=encoder_input_image,
        encoder_input_tokens=encoder_input_tokens,
        frozen_base=frozen_base,
    )
    base_query = self.dropout(base_query, deterministic=not train)
    base_vals = self.dropout(base_vals, deterministic=not train)
    if debug:
      out_dict['base_query'] = base_query
      out_dict['base_masks'] = base_masks
    corpus_scores = jax.nn.softmax(self.dataset_gate(base_query), axis=-1)
    out_dict['corpus_scores'] = corpus_scores
    if no_memory:
      fused_emb, fused_mask, attn_weights_all_layers = self.fusion_encoder(
          fused_input_embs=base_vals, fused_mask=base_masks, use_dropout=train
      )  # B×(I+N)×d
    else:
      if use_memory:
        detached_query = jax.lax.stop_gradient(base_query)
        if retrieve_local:
          (
              topk_scores,
              ret_memory,
              ret_data,
              local_topk_ids,
              global_topk_ids,
              global_corpus_ids,
          ) = self._dist_mips_local(
              query=detached_query,
              corpus_scores=corpus_scores,
              local_device_id=device_id,
          )
        else:
          (
              topk_scores,
              ret_memory,
              ret_data,
              local_topk_ids,
              global_topk_ids,
              global_corpus_ids,
          ) = self._dist_mips_across(
              query=detached_query,
              corpus_scores=corpus_scores,
              local_device_id=device_id,
          )
        out_dict['topk_scores'] = topk_scores

        # encode the retrieved data
        retr_keys, retr_vals, retr_masks, _, disentangle_reg = (
            self.encode_topk_knowledge(
                bsz=bsz,
                retr_images=ret_data['image'],
                retr_texts=ret_data['text_tokens'],
                train=train,
                random_drop_image=False,
                frozen_base=frozen_base,
            )
        )

        global_corpus_scores = layers.batch_index_select(
            corpus_scores, global_corpus_ids
        )

        if debug:
          out_dict['detached_query'] = detached_query
          out_dict['global_corpus_scores'] = global_corpus_scores
          out_dict['global_corpus_ids'] = global_corpus_ids
          out_dict['local_topk_ids'] = local_topk_ids
          out_dict['global_topk_ids'] = global_topk_ids
          out_dict['retr_keys'] = retr_keys
          out_dict['retr_masks'] = retr_masks
          out_dict['base_vals'] = base_vals
          out_dict['retr_vals'] = retr_vals

        out_dict['retr_data'] = ret_data
        out_dict['base_norm'] = layers.l2_norm(base_vals).mean()
        out_dict['data_norm'] = layers.l2_norm(retr_vals).mean()
        out_dict['vals_norm'] = layers.l2_norm(ret_memory['values'][0]).mean()
        out_dict['gap'] = jnp.abs(
            1 - jnp.divide(out_dict['data_norm'], out_dict['base_norm'])
        )

        if train and retr_texts is not None and use_psudo_retr:
          logging.info('global keys!!!')
          ground_truth_keys, ground_truth_vals, _, _, _ = self.encode_knowledge(
              retr_texts=retr_texts,
              retr_images=retr_images,
              bsz=bsz,
              train=train,
              random_drop_image=True,
              frozen_base=frozen_base,
          )
          global_keys = jnp.concatenate(
              jax.lax.all_gather(
                  x=ground_truth_keys, axis_name='batch', axis=0
              ),
              axis=0,
          )
          logging.info(global_keys.shape)
          inbatch_sim = jax.lax.dot(base_query, global_keys.transpose())
          out_dict['inbatch_sim'] = inbatch_sim
          if debug:
            out_dict['global_keys'] = global_keys
            out_dict['ground_truth_keys'] = ground_truth_keys
            out_dict['ground_truth_vals'] = ground_truth_vals
          # replace retrieved knowledge as ground-truth ones for stablization.
          k = retr_keys.shape[1]
          ground_truth_keys = jnp.repeat(
              jnp.expand_dims(ground_truth_keys, axis=1), axis=1, repeats=k
          )
          ground_truth_vals = jnp.repeat(
              jnp.expand_dims(ground_truth_vals, axis=1), axis=1, repeats=k
          )
          replace_mask = jax.random.bernoulli(
              self.make_rng('dropout'), p=0.02, shape=(bsz, 1, 1)
          )
          keys_mask = jnp.broadcast_to(replace_mask, retr_keys.shape)
          retr_keys = jax.lax.select(keys_mask, ground_truth_keys, retr_keys)
          vals_mask = jnp.broadcast_to(
              jnp.expand_dims(replace_mask, axis=-1), retr_vals.shape
          )
          retr_vals = jax.lax.select(vals_mask, ground_truth_vals, retr_vals)

        logging.info('Concat memory and data!!!')
        logging.info(retr_keys.shape)
        logging.info(ret_memory['keys'].shape)
        logging.info(global_corpus_scores.shape)
        # concat retrieved memory (90%) with re-encoded ones (10%)

        retr_keys = jnp.concatenate([retr_keys, ret_memory['keys']], axis=1)
        retr_keys = retr_keys * jnp.expand_dims(global_corpus_scores, axis=-1)
        retr_vals = jnp.concatenate([retr_vals, ret_memory['values']], axis=1)
        retr_masks = jnp.concatenate([retr_masks, ret_memory['masks']], axis=1)
      elif retr_texts is not None:
        retr_keys, retr_vals, retr_masks, _, disentangle_reg = (
            self.encode_topk_knowledge(
                bsz=bsz,
                retr_images=jnp.expand_dims(retr_images, axis=1),
                retr_texts=jnp.expand_dims(retr_texts, axis=1),
                train=train,
                random_drop_image=False,
            )
        )
      else:
        retr_keys, retr_vals, retr_masks, _, disentangle_reg = (
            self.encode_topk_knowledge(
                bsz=bsz,
                retr_images=jnp.expand_dims(encoder_input_image, axis=1),
                retr_texts=jnp.expand_dims(encoder_input_tokens, axis=1),
                train=train,
                random_drop_image=False,
            )
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
      out_dict['disentangle_reg'] = jnp.mean(disentangle_reg)
      out_dict['retr_scores'] = retr_scores

    out_dict['fused_emb'] = fused_emb
    out_dict['fused_mask'] = fused_mask
    logging.info('fused_emb.shape')
    logging.info(fused_emb.shape)
    out_dict['attn_weights_all_layers'] = attn_weights_all_layers

    if not only_encode:
      # decode: generate decoding results.
      out_dict['predicted_logits'] = self.t5_decode(
          encoded=fused_emb,
          encoder_input_tokens=fused_mask,
          decoder_input_tokens=decoder_input_tokens,
          decoder_target_tokens=decoder_target_tokens,
          enable_dropout=train,
          decode=decode,
          max_decode_length=max_decode_length,
      )
    return out_dict


class KnowledgeFIDModel(base_model.BaseModel):
  """FID model with a retrieval module over a knowledge memory."""

  def __init__(
      self,
      config: Optional[ml_collections.ConfigDict],
      dataset_meta_data: Dict[str, Any],
      kb_datasets: Dict[str, dataset_utils.Dataset],
  ) -> None:
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.retr_k = self.config.model.retr_k
    self.retr_data_ratio = self.config.model.retr_data_ratio
    n_device = jax.device_count()
    self.data_k = int(np.ceil(self.retr_k * self.retr_data_ratio))
    device_per_axis = jax.local_device_count()
    if n_device < device_per_axis:
      self.axis_index_groups = None
      self.across_index_groups = None
    else:
      self.axis_index_groups = np.arange(n_device).reshape(
          [n_device // device_per_axis, device_per_axis]
      )
      self.across_index_groups = self.axis_index_groups.T.tolist()
      self.axis_index_groups = self.axis_index_groups.tolist()
    logging.info('axis_index_groups')
    logging.info(self.axis_index_groups)
    logging.info(self.across_index_groups)
    local_kb.initialize(kb_datasets=kb_datasets)
    self.flax_model = self.build_flax_model()

  def build_flax_model(self) -> nn.Module:
    return KnowledgeFIDModule(
        self.config.model,
        retr_k=self.retr_k,
        data_k=self.data_k,
        axis_index_groups=self.axis_index_groups,
        across_index_groups=self.across_index_groups,
    )

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
    model_config = self.config.model
    gen_loss = losses.nll_loss(
        targets=batch['decoder_target_tokens'],
        pred=output['predicted_logits'],
        target_masks=batch['decoder_target_tokens'] > 0,
        label_smoothing=self.config.model.get('label_smoothing'),
    )
    loss_dict = {'gen_loss': gen_loss}

    if 'inbatch_sim' in output:
      score_matrix = output['inbatch_sim']
      bsz = score_matrix.shape[0]
      labels = jnp.arange(bsz) + bsz * jax.lax.axis_index(axis_name='batch')
      contra_loss = losses.nll_loss(
          pred=score_matrix / self.config.model.get('temperature'),
          targets=labels,
      )
      loss_dict['contra_loss'] = contra_loss
      r = model_config.retrieval_ratio
      loss = gen_loss * (1 - r) + contra_loss * r
      accs = jnp.equal(jnp.argmax(score_matrix, axis=1), labels)
      loss_dict['contra_accs'] = accs
    else:
      loss_dict['contra_loss'] = 0.0
      loss_dict['contra_accs'] = 0.0
      loss = gen_loss

    if 'disentangle' in model_config and 'disentangle_reg' in output:
      loss += output['disentangle_reg'] * 1e-2
    if 'gap' in model_config and 'gap' in output:
      loss += output['gap'] * 1e-4
    loss_dict['total_loss'] = loss
    return loss_dict

  def get_metrics_fn(self, split: Optional[str] = None) -> Any:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(outputs,
      batch)```
    """
    return metrics.token_accuracy

  def get_vqa_metrics(
      self,
      logits: jnp.ndarray,
      batch: constants.JTensorDict,
      split: Optional[str] = None,
  ) -> dict[str, float]:
    """Returns the VQA Accuracy for the validation / test set.

    Args:
      logits: Output of model in shape [B, L, C].
      batch: Batch of data that has 'decoder_target' as ground-truth.
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: VQA accuracy```
    """

    return metrics.vqa_accuracy(logits, batch)

  def single_decode_step(
      self,
      decoding_state: decoding.DecodingState,
      variables: constants.PyTree,
      encoded_inputs: jnp.ndarray,
      input_masks: jnp.ndarray,
      max_decode_length: int,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Single autoregressive decode step with caching."""
    flat_ids = decoding_state.cur_token
    flat_cache = decoding_state.cache
    # flat_ids: [batch * beam, seq_len=1]
    # cache is expanded inside beam_search to become flat_cache
    # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
    # flat_logits: [batch * beam, seq_len=1, vocab]
    flat_logits, new_vars = self.flax_model.apply(
        {'cache': flat_cache, **variables},
        encoded=encoded_inputs,
        encoder_input_tokens=input_masks,
        decoder_input_tokens=flat_ids,
        decoder_target_tokens=flat_ids,
        decode=True,
        enable_dropout=False,
        max_decode_length=max_decode_length,
        mutable=['cache'],
        method=self.flax_model.t5_decode,
    )
    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  def apply_with_autoregressive_decoding(
      self,
      variables: constants.PyTree,
      decoder_input_tokens: jnp.ndarray,
      decoder_target_tokens: jnp.ndarray,
      encoder_input_image: Optional[jnp.ndarray] = None,
      encoder_input_tokens: Optional[jnp.ndarray] = None,
      num_decodes: int = 1,
      debug: bool = False,
      beam_search: bool = True,
      decoder_params: Optional[dict[str, Any]] = None,
      return_all_decodes: bool = False,
      use_memory=False,
      retrieve_local=False,
      **args
  ):
    """Apply inference with autoregressive decoding.

    Apply t5x autoregressive decoding with cache using either their
    beam_search or temperature_sample decoding technique.

    Args:
      variables: variables of the models.
      decoder_input_tokens:  # B×O.
      decoder_target_tokens:  # B×O.
      encoder_input_image:  # B×W×H×3.
      encoder_input_tokens: # B×I.
      num_decodes: number of outputs generated per input for the decode search.
      debug: Whether in debug mode or not.
      beam_search: If True, do beam search. If False, do temperature sampling.
      decoder_params: Additional decoding parameters. These provide additional
        parameters to beam_search or temperature_sample (see decoder module).
      return_all_decodes: If True, return all decodes. Otherwise only return the
        top scored decoding.
      use_memory: whether use on-device memory.
      retrieve_local: whether only retrieve in local host or across hosts.
      **args: other possible arguments.

    Returns:
      logits array from the final decoder.
    """
    # Prepare zeroed-out autoregressive cache.
    _, model_state_with_cache = self.flax_model.apply(
        variables=variables,
        encoder_input_image=encoder_input_image,
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        train=False,
        only_encode=False,
        decode=True,
        mutable=['cache'],
        debug=debug,
        use_memory=use_memory,
        retrieve_local=retrieve_local,
    )

    # Call model to get the features consumed by the decoder. Skip the
    # the decoding part itself.
    out_dict = self.flax_model.apply(
        variables=variables,
        encoder_input_image=encoder_input_image,
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        train=False,
        only_encode=True,
        debug=debug,
        use_memory=use_memory,
        retrieve_local=retrieve_local,
    )
    retr_top_image = out_dict['retr_data']['image'][:, 0]

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    beam_expand_fn = functools.partial(
        decoding.flat_batch_beam_expand, beam_size=num_decodes
    )
    encoded_inputs = jax.tree_util.tree_map(
        beam_expand_fn, out_dict['fused_emb']
    )
    encoded_masks = jax.tree_util.tree_map(
        beam_expand_fn, out_dict['fused_mask']
    )
    bsz = decoder_input_tokens.shape[0]
    max_decode_length = decoder_input_tokens.shape[-1]
    # Define the token2logit function for a single decoding step.
    tokens_ids_to_logits = functools.partial(
        self.single_decode_step,
        variables=variables,
        encoded_inputs=encoded_inputs,
        input_masks=encoded_masks,
        max_decode_length=decoder_input_tokens.shape[-1],
    )

    if decoder_params is None:
      decoder_params = {}
    # For beam search, `decoder_prompt_inputs` is only used to obtain batch size
    # and max decode length information. For temperature sampling,
    # `decod_prompt_inputs` will be filled with the sampled ids.
    decoder_prompt_inputs = jnp.zeros([bsz, max_decode_length - 1])
    bos_inputs = jnp.ones([bsz, 1]) * data_utils.BOS_ID
    decoder_prompt_inputs = jnp.concatenate(
        (bos_inputs, decoder_prompt_inputs), axis=-1, dtype=jnp.int32
    )
    if beam_search:
      decodes, scores = decoding.beam_search(
          inputs=decoder_prompt_inputs,
          cache=model_state_with_cache['cache'],
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=data_utils.EOS_ID,
          num_decodes=num_decodes,
          cache_offset=0,
          **decoder_params
      )
    else:
      decodes, scores = decoding.temperature_sample(
          inputs=decoder_prompt_inputs,
          cache=model_state_with_cache['cache'],
          tokens_to_logits=tokens_ids_to_logits,
          eos_id=data_utils.EOS_ID,
          num_decodes=num_decodes,
          cache_offset=0,
          initial_index=jnp.zeros([bsz], dtype=jnp.int32),
          **decoder_params
      )
    if return_all_decodes:
      return decodes, scores, retr_top_image
    else:
      return decodes[:, -1, :], scores[:, -1], retr_top_image
