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

"""Tests loading of OWL-ViT checkpoints for real configs.."""
import functools
import inspect
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models


class CheckpointLoadingTest(parameterized.TestCase):
  """Tests that checkpoints can be loaded."""

  @parameterized.named_parameters(
      *inspect.getmembers(configs, inspect.ismodule)
  )
  def test_checkpoint_loading(self, config_module):
    """Tests that real checkpoints can be loaded and used with the model."""
    # We test the canonical checkpoint if there is one:
    try:
      config = config_module.get_config(init_mode='canonical_checkpoint')
    except TypeError:
      config = config_module.get_config()

    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias,
    )

    # Parameter initialization:
    batch_size = 8
    img_size = config.dataset_configs.input_size
    num_queries = 10
    seq_len = config.dataset_configs.max_query_length
    images = jnp.ones((batch_size, img_size, img_size, 3))
    texts = jnp.ones((batch_size, num_queries, seq_len), dtype=jnp.int32)
    init = functools.partial(module.init, train=False)
    init_params = jax.eval_shape(init, jax.random.PRNGKey(0), images, texts)
    params = module.bind({}).load(init_params, config.init_from)

    # Test running the model with the parameters:
    fn = functools.partial(module.apply, train=False)
    out = jax.eval_shape(fn, {'params': params}, images, texts)

    self.assertContainsSubset({'pred_boxes', 'pred_logits'}, out.keys())


if __name__ == '__main__':
  absltest.main()
