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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from scenic.projects.pointcloud import models


# pylint: disable=invalid-name

IN_DIM = 3
OUT_DIM = 1024
DIM = 3
FEATURE_DIM = 12
BATCH = 10
NB_POINTS = 1000
SEED = 41


class PerformerModelTest(parameterized.TestCase):

  def test_regular_transformer_encoder_result(self):
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM, feature_dim=FEATURE_DIM, attention_fn_configs=None
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_regular_performer_softmax_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'nomask',
        'kernel_transformation': 'softmax',
        'num_features': 64,
        'rpe_method': None,
        'num_realizations': 10,
        'num_sines': 1,
        'use_random_projections': True,
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_regular_performer_relu_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'nomask',
        'kernel_transformation': 'relu',
        'num_features': 0,
        'rpe_method': None,
        'num_realizations': 10,
        'num_sines': 1,
        'use_random_projections': False,
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_performer_softmax_fftmasked_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'fftmasked',
        'kernel_transformation': 'softmax',
        'num_features': 64,
        'rpe_method': None,
        'num_realizations': 10,
        'num_sines': 1,
        'use_random_projections': True,
        'seed': 41
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_performer_relu_fftmasked_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'fftmasked',
        'kernel_transformation': 'relu',
        'num_features': 0,
        'rpe_method': None,
        'num_realizations': 10,
        'num_sines': 1,
        'use_random_projections': False,
        'seed': 41
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_performer_softmax_sharpmasked_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'sharpmasked',
        'kernel_transformation': 'softmax',
        'num_features': 64,
        'rpe_method': None,
        'num_realizations': 10,
        'num_sines': 1,
        'use_random_projections': True,
        'seed': 41
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_performer_relu_sharpmasked_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'sharpmasked',
        'kernel_transformation': 'relu',
        'num_features': 0,
        'rpe_method': None,
        'num_realizations': 10,
        'num_sines': 1,
        'use_random_projections': False,
        'seed': 41
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))

  def test_performer_psuedolocal_masking_encoder_result(self):
    attention_fn_configs = dict()
    attention_fn_configs['attention_kind'] = 'performer'
    attention_fn_configs['performer'] = {
        'masking_type': 'pseudolocal',
        'rf_type': 'regular',
        'num_features': 128,
    }
    rng_key = jax.random.PRNGKey(SEED)
    inputs = jax.random.normal(key=rng_key, shape=(BATCH, NB_POINTS, DIM))
    pct_encoder = models.PointCloudTransformerEncoder(
        in_dim=IN_DIM,
        feature_dim=FEATURE_DIM,
        attention_fn_configs=attention_fn_configs,
    )
    variables = pct_encoder.init(rng_key, inputs)
    result = pct_encoder.apply(variables, inputs)
    self.assertEqual(result.shape, (BATCH, NB_POINTS, OUT_DIM))


if __name__ == '__main__':
  absltest.main()
