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

"""Tests for encoders."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import ml_collections
import numpy as np
from scenic.projects.unloc import encoders


class EncodersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temporal_encoding_config = ml_collections.ConfigDict({
        'method': '3d_conv',
        'kernel_init_method': 'central_frame_initializer',
    })
    self.image_encoder_config = ml_collections.ConfigDict({
        'num_layers': 2,
        'features': 8,
        'patches': ml_collections.ConfigDict({
            'size': (4, 4, 1),
        }),
        'num_heads': 2,
        'classifier': 'token',
    })

  @parameterized.parameters(
      ('token', 'logits', (2, 10)),
      ('token', 'pre_logits', (2, 8)),
      ('token', 'temporal_tokens', (2, 2, 8)),
  )
  def test_clip_video_tower(self, image_encoder_classifier, final_endpoint,
                            expected_output_shape):
    rng = random.PRNGKey(0)
    inputs = np.ones((2, 2, 8, 8, 3))
    self.image_encoder_config.classifier = image_encoder_classifier
    output, _ = encoders.ClipVideoTower(
        image_encoder_config=self.image_encoder_config,
        temporal_encoding_config=self.temporal_encoding_config,
        temporal_encoder_config=None,
        num_classes=10,
        final_endpoint=final_endpoint,
    ).init_with_output(rng, inputs, train=False, debug=False)
    self.assertTupleEqual(output.shape, expected_output_shape)

  def test_clip_text_encoder(self):
    rng = random.PRNGKey(0)
    inputs = {
        'input_word_ids': np.ones((2, 10), np.int32),
        'input_type_ids': np.zeros((2, 10), np.int32),
        'input_mask': np.ones((2, 10), np.int32),
    }
    output, _ = encoders.ClipTextEncoder(
        vocab_size=100, num_layers=2, hidden_size=8, num_heads=2
    ).init_with_output(rng, inputs, train=False, debug=False)
    self.assertTupleEqual(output.shape, (2, 10, 8))

  def test_pass_through_encoder(self):
    rng = random.PRNGKey(0)
    inputs = np.ones((2, 10), dtype=np.float32)
    output, _ = encoders.PassThroughEncoder().init_with_output(
        rng, inputs, train=False, debug=False
    )
    self.assertTrue(np.array_equal(output, inputs))


if __name__ == '__main__':
  absltest.main()
