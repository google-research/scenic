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

"""Tests for model.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
from flax import traverse_util
from flax.core import unfreeze
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.baselines.detr import model


class DETRModulesTest(parameterized.TestCase):
  """Tests for detr model.py."""

  def test_input_pos_embedding_learned(self):
    """Tests InputPosEmbeddingLearned."""
    rng = random.PRNGKey(0)
    inputs_shape = (8, 32, 19, 64)

    positional_embedding_def = model.InputPosEmbeddingLearned(
        inputs_shape=inputs_shape, hidden_dim=64, max_h_w=50)
    pos, _ = positional_embedding_def.init_with_output(rng)
    # test output shape
    self.assertEqual(pos.shape, (1, 32 * 19, 64))

  def test_input_pos_embedding_sine(self):
    """Tests InputPosEmbeddingSine."""
    rng = random.PRNGKey(0)
    b, h, w, hidden = (8, 32, 19, 64)
    padding_mask = np.zeros((b, h, w))
    padding_mask[:, 16:, :] = 0
    padding_mask[:, :, 16:] = 0

    positional_embedding_def = model.InputPosEmbeddingSine(hidden_dim=hidden)
    pos, _ = positional_embedding_def.init_with_output(rng, padding_mask)
    # test output shape
    self.assertEqual(pos.shape, (b, h * w, hidden))

  def test_query_pos_embedding(self):
    """Tests QueryPosEmbedding."""
    rng = random.PRNGKey(0)
    positional_embedding_def = model.QueryPosEmbedding(
        hidden_dim=64, num_queries=100)
    query_pos, _ = positional_embedding_def.init_with_output(rng)
    # test output shape
    self.assertEqual(query_pos.shape, (1, 100, 64))

  @parameterized.named_parameters(
      ('test_without_intermediate', False, (8, 100, 64)),
      ('test_with_intermediate', True, (6, 8, 100, 64)),
  )
  def test_detr_transformer_output_shape(self, return_intermediate,
                                         expected_output_shape):
    """Test DETRTransformer output shape."""
    rng = random.PRNGKey(0)
    inputs_shape = (8, 20 * 20, 64)
    num_query_objects = 100
    num_decoder_layers = 6
    inputs = jnp.array(np.random.normal(size=inputs_shape))
    query_pos_emb = jnp.array(np.random.normal(size=(1, num_query_objects, 64)))

    # test output shape of DETR Transformer model
    detr_transformer_def = model.DETRTransformer(
        num_queries=num_query_objects,
        qkv_dim=64,
        num_heads=4,
        num_decoder_layers=num_decoder_layers,
        return_intermediate_dec=return_intermediate)

    (outputs, _), _ = detr_transformer_def.init_with_output(
        rng, inputs, query_pos_emb=query_pos_emb)
    self.assertEqual(outputs.shape, expected_output_shape)


class DETRModelTest(parameterized.TestCase):
  """Test DETRModel."""

  def setUp(self):
    super(DETRModelTest, self).setUp()

    self.num_classes = 5
    self.input_shape = (3, 128, 128, 3)

    # create and initialize the model
    model_cls = model.DETRModel
    config = ml_collections.ConfigDict(
        dict(
            hidden_dim=32,
            num_queries=8,
            query_emb_size=None,
            transformer_num_heads=2,
            transformer_num_encoder_layers=1,
            transformer_qkv_dim=32,
            transformer_mlp_dim=32,
            transformer_normalize_before=False,
            backbone_num_filters=32,
            backbone_num_layers=14,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            class_loss_coef=1.0,
            bbox_loss_coef=1.0,
            giou_loss_coef=1.0,
            eos_coef=1.0,
            aux_loss=False,
            transformer_num_decoder_layers=3,
        ))
    self.model = model_cls(
        config=config,
        dataset_meta_data={
            'num_classes': self.num_classes,
            'target_is_onehot': False,
        })
    rng = random.PRNGKey(0)
    initial_params = self.model.flax_model.init(
        rng, jnp.zeros(self.input_shape, jnp.float32), train=False)
    flax_model = functools.partial(
        self.model.flax_model.apply,
        initial_params,
        mutable=['intermediates'],
        capture_intermediates=lambda mdl, _: mdl.name == 'attn_weights')

    # a fake batch with 3 examples
    self.batch = {
        'inputs':
            jnp.array(np.random.normal(size=self.input_shape)
                     ).astype(jnp.float32),
        'padding_mask':
            jnp.array(np.random.normal(size=self.input_shape[:-1])
                     ).astype(jnp.float32),
        'label': {
            'labels':
                jnp.array(
                    np.random.randint(
                        self.num_classes,
                        size=(3, self.model.config.num_queries))),
            'boxes':
                jnp.array(
                    np.random.uniform(
                        size=(3, self.model.config.num_queries, 4),
                        low=0.0,
                        high=1.0),
                    dtype=jnp.float32),
        }
    }

    self.outputs, self.variables = flax_model(
        self.batch['inputs'],
        padding_mask=self.batch['padding_mask'],
        train=False)

    seq = np.arange(self.model.config.num_queries, dtype=np.int32)
    seq_rev = seq[::-1]
    seq_21 = np.concatenate([
        seq[self.model.config.num_queries // 2:],
        seq[:self.model.config.num_queries // 2]
    ])
    self.indices = jnp.array([(seq, seq_rev), (seq_rev, seq), (seq, seq_21)])

  def is_valid(self, t):
    """Helper function to assert that tensor `t` does not have `nan`, `inf`."""
    self.assertFalse(jnp.isnan(t).any(), msg=f'Found nan\'s in {t}')
    self.assertFalse(jnp.isinf(t).any(), msg=f'Found inf\'s in {t}')

  def is_valid_loss(self, loss):
    """Helper function to assert that `loss` is of shape [] and `is_valid`."""
    self.assertSequenceEqual(loss.shape, [])
    self.is_valid(loss)

  @parameterized.named_parameters(
      ('without_log', False, ['loss_class']),
      ('with_log', True,
       ['loss_class', 'class_accuracy', 'class_accuracy_not_pad']))
  def test_labels_losses_and_metrics(self, log, metrics_key):
    """Test loss_labels by checking its output's dictionary format.

    Args:
      log: bool; Whether do logging or not in labels_losses_and_metrics.
      metrics_key: list; Expected metric keys.
    """

    def f_to_pmap(outputs, batch):
      return self.model.labels_losses_and_metrics(
          outputs, batch, indices=self.indices, log=log)

    # test loss function in the pmapped setup
    labels_lm_pmapped = jax.pmap(f_to_pmap, axis_name='batch')

    outputs, batch = (jax_utils.replicate(self.outputs),
                      jax_utils.replicate(self.batch))

    losses, metrics = labels_lm_pmapped(outputs, batch)
    losses = jax_utils.unreplicate(losses)
    metrics = jax_utils.unreplicate(metrics)
    self.assertSameElements(losses.keys(), ['loss_class'])
    self.is_valid_loss(losses['loss_class'])
    self.assertSameElements(metrics.keys(), metrics_key)
    for mk in metrics_key:
      self.is_valid(metrics[mk][0])
      self.is_valid(metrics[mk][1])

  def test_boxes_losses_and_metrics(self):
    """Test loss_boxes by checking its output's dictionary format."""

    def f_to_pmap(outputs, batch):
      return self.model.boxes_losses_and_metrics(
          outputs, batch, indices=self.indices)

    # test loss function in the pmapped setup
    boxes_lm_pmapped = jax.pmap(f_to_pmap, axis_name='batch')

    outputs_replicate, batch_replicate = (jax_utils.replicate(self.outputs),
                                          jax_utils.replicate(self.batch))

    losses, metrics = boxes_lm_pmapped(outputs_replicate, batch_replicate)
    losses = jax_utils.unreplicate(losses)
    metrics = jax_utils.unreplicate(metrics)

    self.assertSameElements(losses.keys(), ['loss_bbox', 'loss_giou'])
    self.is_valid_loss(losses['loss_bbox'])
    self.is_valid_loss(losses['loss_giou'])

    self.assertSameElements(metrics.keys(), ['loss_bbox', 'loss_giou'])
    for i in range(2):  # metric and its normalizer
      self.is_valid(metrics['loss_bbox'][i])
      self.is_valid(metrics['loss_giou'][i])

  def test_intermediates(self):
    """Test the capture of intermediates."""

    intermediates = self.variables['intermediates']
    keys = traverse_util.flatten_dict(unfreeze(intermediates)).keys()
    actual = ['/'.join(key) for key in keys]
    expected = [
        'DETRTransformer_0/decoder/decoderblock_0/MultiHeadDotProductAttention_0/attn_weights/__call__',
        'DETRTransformer_0/decoder/decoderblock_0/MultiHeadDotProductAttention_1/attn_weights/__call__',
        'DETRTransformer_0/decoder/decoderblock_1/MultiHeadDotProductAttention_0/attn_weights/__call__',
        'DETRTransformer_0/decoder/decoderblock_1/MultiHeadDotProductAttention_1/attn_weights/__call__',
        'DETRTransformer_0/decoder/decoderblock_2/MultiHeadDotProductAttention_0/attn_weights/__call__',
        'DETRTransformer_0/decoder/decoderblock_2/MultiHeadDotProductAttention_1/attn_weights/__call__',
        'DETRTransformer_0/encoder/encoderblock_0/MultiHeadDotProductAttention_0/attn_weights/__call__',
    ]
    self.assertSameElements(expected, actual)


class DETRModelTestWithAuxLoss(parameterized.TestCase):
  """Test DETRModel with auxilary loss."""

  def setUp(self):
    super(DETRModelTestWithAuxLoss, self).setUp()

    self.num_classes = 5
    self.input_shape = (3, 128, 128, 3)
    config = ml_collections.ConfigDict(
        dict(
            hidden_dim=32,
            num_queries=8,
            query_emb_size=None,
            transformer_num_heads=2,
            transformer_num_encoder_layers=1,
            transformer_qkv_dim=32,
            transformer_mlp_dim=32,
            transformer_normalize_before=False,
            backbone_num_filters=32,
            backbone_num_layers=14,
            dropout_rate=0.1,
            class_loss_coef=1.0,
            bbox_loss_coef=1.0,
            giou_loss_coef=1.0,
            eos_coef=1.0,
            # for this test:
            aux_loss=True,
            transformer_num_decoder_layers=3,
        ))

    # create and initialize the model
    model_cls = model.DETRModel
    self.model = model_cls(
        config=config,
        dataset_meta_data={
            'num_classes': self.num_classes,
            'target_is_onehot': False,
        })

    rng = random.PRNGKey(0)
    initial_params = self.model.flax_model.init(
        rng, jnp.zeros(self.input_shape, jnp.float32), train=False)
    flax_model = functools.partial(self.model.flax_model.apply, initial_params)

    # a fake batch with 3 examples
    self.batch = {
        'inputs':
            jnp.array(np.random.normal(size=self.input_shape)
                     ).astype(jnp.float32),
        'padding_mask':
            jnp.array(np.random.normal(size=self.input_shape[:-1])
                     ).astype(jnp.float32),
        'label': {
            'labels':
                jnp.array(
                    np.random.randint(
                        self.num_classes,
                        size=(3, self.model.config.num_queries))),
            'boxes':
                jnp.array(
                    np.random.uniform(
                        size=(3, self.model.config.num_queries, 4),
                        low=0.0,
                        high=1.0),
                    dtype=jnp.float32),
        }
    }
    self.outputs = flax_model(
        self.batch['inputs'],
        padding_mask=self.batch['padding_mask'],
        train=False)

    seq = np.arange(self.model.config.num_queries, dtype=np.int32)
    seq_rev = seq[::-1]
    seq_21 = np.concatenate([
        seq[self.model.config.num_queries // 2:],
        seq[:self.model.config.num_queries // 2]
    ])
    self.indices = jnp.array([(seq, seq_rev), (seq_rev, seq), (seq, seq_21)])

  def test_loss_function(self):
    """Test loss_function by checking its output's dictionary format."""

    # test loss function in the pmapped setup
    loss_function_pmapped = jax.pmap(
        self.model.loss_function, axis_name='batch')

    matches = jax_utils.replicate(
        # fake matching for the  final output +  2 aux outputs
        [self.indices] * 3)
    outputs_replicated, batch_replicated = (jax_utils.replicate(self.outputs),
                                            jax_utils.replicate(self.batch))
    total_loss, metrics_dict = loss_function_pmapped(
        outputs_replicated, batch_replicated, matches=matches)

    total_loss, metrics_dict = (jax_utils.unreplicate(total_loss),
                                jax_utils.unreplicate(metrics_dict))

    # collect what keys we expect to find in the metrics_dict
    base = [
        'class_accuracy',
        'loss_class',
        'loss_bbox',
        'loss_giou',
        'total_loss',
        'loss_class_aux_0',
        'loss_bbox_aux_0',
        'loss_giou_aux_0',
        'loss_class_aux_1',
        'loss_bbox_aux_1',
        'loss_giou_aux_1',
        'class_accuracy_not_pad',
    ]
    base_unscaled = []
    for b in base:
      if b.split('_aux_')[0] in self.model.loss_terms_weights.keys():
        base_unscaled.append(b + '_unscaled')
      else:
        base_unscaled.append(b)
    base_scaled = [
        'loss_class',
        'loss_bbox',
        'loss_giou',
        'loss_class_aux_0',
        'loss_bbox_aux_0',
        'loss_giou_aux_0',
        'loss_class_aux_1',
        'loss_bbox_aux_1',
        'loss_giou_aux_1',
    ]
    expected_metrics_keys = base_unscaled + base_scaled
    self.assertSameElements(expected_metrics_keys, metrics_dict.keys())

    # because weight decay is not used, the following must hold
    object_detection_loss = 0
    for k in metrics_dict.keys():
      b = k.split('_aux_')[0]
      # if this loss going to be included in the total object dtection loss
      if '_unscaled' not in k and b in self.model.loss_terms_weights.keys():
        # get the normalizer for this loss
        object_detection_loss += (
            # already scaled loss term / loss term normalizer
            metrics_dict[k][0] / metrics_dict[k][1])
    self.assertAlmostEqual(total_loss, object_detection_loss, places=3)


if __name__ == '__main__':
  absltest.main()
