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

"""Tests for FlexIO input pipeline."""

from absl.testing import absltest
from absl.testing import parameterized
from grand_vision.preprocessing import image_ops
from grand_vision.preprocessing import modalities
import jax
import ml_collections
from scenic.dataset_lib.flexio import flexio
import tensorflow as tf


D = ml_collections.ConfigDict


class InputPipelineTest(tf.test.TestCase, parameterized.TestCase):
  """Test cases for FlexIO input pipeline."""

  @parameterized.named_parameters(
      ('coco_coco', 'coco', 'coco'),
  )
  def test_tfds_datasets(self, train_tfds_name, eval_tfds_name):
    """Test TFDS dataset loading."""
    dataset_configs = D({
        'train': {
            'sources': [D({
                'source': 'tfds',
                'tfds_name': train_tfds_name,
                'split': 'train',
                'shuffle_buffer_size': 2,
                'cache': False,
                'preproc_spec': 'decode_coco_example|crop_or_pad(64, 16)',
            })],
            'preproc_spec': 'crop_or_pad_meta_data(16, 16)',
        },
        'eval': {
            'sources': [D({
                'source': 'tfds',
                'tfds_name': eval_tfds_name,
                'split': 'validation',
                'shuffle_buffer_size': 1,
                'cache': False,
                'preproc_spec': 'decode_coco_example',
            })],
            'preproc_spec': ('central_crop(64)'
                             '|crop_or_pad(64, 16)'
                             '|crop_or_pad_meta_data(16, 16)'),
        },
        'pp_libs': [  # We override the default ops.
            'grand_vision.preprocessing.image_ops']
    })
    rng = jax.random.PRNGKey(0)
    num_devices = jax.local_device_count()
    ds = flexio.get_dataset(
        batch_size=8,
        eval_batch_size=8,
        num_shards=num_devices,
        rng=rng,
        dataset_configs=dataset_configs)
    per_device = 8 // num_devices
    prefix_shape = (num_devices, per_device)
    expected_shapes = {
        modalities.ANNOTATION_ID: prefix_shape + (16,),
        modalities.AREA: prefix_shape + (16,),
        modalities.BOXES: prefix_shape + (16, 4),
        modalities.CROWD: prefix_shape + (16,),
        modalities.IMAGE: prefix_shape + (64, 64, 3),
        modalities.IMAGE_ID: prefix_shape,
        modalities.IMAGE_PADDING_MASK: prefix_shape + (64, 64),
        modalities.INSTANCE_LABELS: prefix_shape + (16,),
        modalities.ORIGINAL_SIZE: prefix_shape + (2,),
        image_ops.SEED_KEY: prefix_shape + (2,)
    }
    train_data = next(ds.train_iter)
    valid_data = next(ds.valid_iter)
    self.assertDictEqual(
        jax.tree_util.tree_map(lambda x: x.shape, train_data), expected_shapes)
    self.assertDictEqual(
        jax.tree_util.tree_map(lambda x: x.shape, valid_data), expected_shapes)


if __name__ == '__main__':
  absltest.main()
