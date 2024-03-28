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

"""Tests for inference."""

from unittest import mock

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b32
from scenic.projects.owl_vit.notebooks import inference


def _mock_tokenize(text, max_len):
  del text
  return np.zeros((max_len,), dtype=np.int32)


class InferenceTest(absltest.TestCase):

  def setUp(self):
    config = clip_b32.get_config(init_mode='canonical_checkpoint')
    config.dataset_configs.input_size = 128
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias)
    rng = jax.random.PRNGKey(0)
    image = jnp.zeros((1, 128, 128, 3), dtype=jnp.float32)
    text_queries = jnp.zeros((1, 5, config.dataset_configs.max_query_length),
                             dtype=jnp.int32)
    variables = module.init(rng, image, text_queries, train=False)
    self.model = inference.Model(config, module, variables)
    self.num_instances = (config.dataset_configs.input_size // 32) ** 2
    super().setUp()

    self.enter_context(
        mock.patch.object(
            target=models.clip_tokenizer,
            attribute='tokenize',
            autospec=True,
            side_effect=_mock_tokenize))

  def test_warm_up(self):
    """Tests that the model can be compiled and run a forward pass."""
    self.model.warm_up()

  def test_preprocess_image(self):
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    processed = self.model.preprocess_image(image)
    self.assertEqual(processed.dtype, np.float32)
    input_size = self.model.config.dataset_configs.input_size
    chex.assert_shape(processed, (input_size, input_size, 3))

  def test_embed_image(self):
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    (image_features, image_class_embeddings,
     pred_boxes) = self.model.embed_image(image)

    self.assertIsInstance(image_features, np.ndarray)
    self.assertIsInstance(image_class_embeddings, np.ndarray)
    self.assertIsInstance(pred_boxes, np.ndarray)

    chex.assert_shape(image_features, (self.num_instances, 768))
    chex.assert_shape(image_class_embeddings, (self.num_instances, 512))
    chex.assert_shape(pred_boxes, (self.num_instances, 4))

  def test_embed_text_queries(self):
    queries = ('query1', 'query2', '')
    query_embeddings = self.model.embed_text_queries(queries)
    self.assertIsInstance(query_embeddings, np.ndarray)
    chex.assert_shape(query_embeddings, (inference.QUERY_PAD_BIN_SIZE, 512))

  def test_embed_image_query(self):
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    box = (0.4, 0.4, 0.5, 0.5)
    query_embedding, box_ind = self.model.embed_image_query(image, box)
    self.assertIsInstance(query_embedding, np.ndarray)
    chex.assert_shape(query_embedding, (512,))
    chex.assert_shape(box_ind, ())

  def test_get_scores(self):
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    query_embeddings = np.zeros((inference.QUERY_PAD_BIN_SIZE, 512))
    num_queries = 3
    top_query_ind, scores = self.model.get_scores(
        image, query_embeddings, num_queries)

    self.assertIsInstance(top_query_ind, np.ndarray)
    self.assertIsInstance(scores, np.ndarray)

    chex.assert_shape(top_query_ind, (self.num_instances,))
    chex.assert_shape(scores, (self.num_instances,))


if __name__ == '__main__':
  absltest.main()
