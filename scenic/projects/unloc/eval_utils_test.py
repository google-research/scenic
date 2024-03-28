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

"""Tests for eval_utils."""

import numpy as np
from scenic.projects.unloc import eval_utils
import tensorflow as tf


class ClassificationTrainUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.batch = {
        'inputs': {
            'rgb': np.zeros((2, 4, 8, 8, 3)),
            'class_names': {
                'input_word_ids': np.ones((2, 10 * 3, 8), dtype=np.int32),
                'input_type_ids': np.zeros((2, 10 * 3, 8), dtype=np.int32),
                'input_mask': np.ones((2, 10 * 3, 8), dtype=np.int32),
            }
        },
        'label': np.zeros((2, 10)),
        'batch_mask': np.ones((2)),
    }
    self.text_emb_batch = {
        'inputs': {
            'rgb': np.zeros((2, 4, 8, 8, 3), dtype=np.float32),
            'class_names': np.ones((2, 10 * 3, 8), dtype=np.float32),
        },
        'label': np.zeros((2, 10)),
        'batch_mask': np.ones((2)),
    }

  def assertDictEqualRecursive(self, actual, expected):
    self.assertEqual(type(actual), type(expected))
    if isinstance(actual, dict):
      self.assertSameElements(actual.keys(), expected.keys())
      for key, _ in expected.items():
        self.assertDictEqualRecursive(actual[key], expected[key])
    elif isinstance(actual, np.ndarray):
      np.testing.assert_allclose(actual, expected)
    else:
      self.assertEqual(actual, expected)

  def test_get_input_batch_from_one_prompt(self):
    clip_input = eval_utils._get_input_batch_from_one_prompt(
        self.batch, num_classes=10, prompt_index=0, crop_index=0, n_clips=2)
    expected_output = {
        'rgb': np.zeros((2, 4, 8, 8, 3)),
        'class_names': {
            'input_word_ids': np.ones((2, 10, 8), dtype=np.int32),
            'input_type_ids': np.zeros((2, 10, 8), dtype=np.int32),
            'input_mask': np.ones((2, 10, 8), dtype=np.int32),
        }
    }
    self.assertDictEqualRecursive(clip_input, expected_output)

  def test_get_input_text_emb_batch_from_one_prompt(self):
    clip_input = eval_utils._get_input_batch_from_one_prompt(
        self.text_emb_batch,
        num_classes=10,
        prompt_index=0,
        crop_index=0,
        n_clips=2)
    expected_output = {
        'rgb': np.zeros((2, 4, 8, 8, 3), dtype=np.float32),
        'class_names': np.ones((2, 10, 8), dtype=np.float32),
    }
    self.assertDictEqualRecursive(clip_input, expected_output)


class MomentRetrievalTrainUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.batch = {
        'num_classes': 1,
        'inputs': {
            'rgb': np.zeros((2, 4, 8, 8, 3)),
            'caption': {
                'input_word_ids': np.ones((2, 3, 10), dtype=np.int32),
                'input_type_ids': np.zeros((2, 3, 10), dtype=np.int32),
                'input_mask': np.ones((2, 3, 10), dtype=np.int32),
            },
            'input_mask': np.ones((2, 4)),
            'caption_mask': np.ones((2, 3)),
        },
        'batch_mask': np.ones((2)),
        'total_frames': np.ones((2), dtype=np.int32),
        'label': np.zeros((2, 3, 4, 1)),
        'displacements': np.zeros((2, 3, 4, 2)),
        'segment_start_index': np.zeros((2, 3), dtype=np.int32),
        'segment_end_index': np.ones((2, 3), dtype=np.int32),
        'segment_start_timestamp': np.zeros((2, 3), dtype=np.int32),
        'segment_end_timestamp': np.ones((2, 3), dtype=np.int32),
    }

  def assertDictEqualRecursive(self, actual, expected):
    self.assertEqual(type(actual), type(expected))
    if isinstance(actual, dict):
      self.assertSameElements(actual.keys(), expected.keys())
      for key, _ in expected.items():
        self.assertDictEqualRecursive(actual[key], expected[key])
    elif isinstance(actual, np.ndarray):
      np.testing.assert_allclose(actual, expected)
    else:
      self.assertEqual(actual, expected)

  def test_get_input_batch_from_one_prompt(self):
    clip_input = eval_utils._get_input_batch_from_one_prompt(
        self.batch, num_classes=1, prompt_index=0, crop_index=0, n_clips=1)
    expected_output = {
        'rgb': np.zeros((1, 4, 8, 8, 3)),
        'caption': {
            'input_word_ids': np.ones((1, 3, 10), dtype=np.int32),
            'input_type_ids': np.zeros((1, 3, 10), dtype=np.int32),
            'input_mask': np.ones((1, 3, 10), dtype=np.int32),
        },
        'input_mask': np.ones((1, 4)),
        'caption_mask': np.ones((1, 3)),
    }
    self.assertDictEqualRecursive(clip_input, expected_output)

if __name__ == '__main__':
  tf.test.main()
