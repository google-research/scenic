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

"""Tests for the linear low-rank (LLR) attention library."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp

from scenic.projects.performer import subquadratic_attention as sat
from scenic.projects.performer import utils as ut

QUERY_RAND_SEED = 143567883590
KEY_RAND_SEED = 847392817892
VALUE_RAND_SEED = 5939874023


class KernelTransformationAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'softmax_positive_rfs',
          sat.softmax_positive_rfs,
      ),
      (
          'softmax_hyper_positive_rfs',
          sat.softmax_hyper_positive_rfs,
      ),
  )
  def test_evaluate_parameter(self, sat_rfs):

    # query -> [batch_size, length, num_heads, features]
    # key -> [batch_size, length, num_heads, features]
    # value -> [batch_size, length, num_heads, features]

    qk_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    nb_random_features = 10000
    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, v_dim)
    query = random.normal(random.PRNGKey(QUERY_RAND_SEED), shape_query)
    key = random.normal(random.PRNGKey(KEY_RAND_SEED), shape_key)
    value = random.normal(random.PRNGKey(VALUE_RAND_SEED), shape_value)
    projection_matrix = ut.get_gaussian_orth_rand_mat(
        random.PRNGKey(0), nb_random_features, qk_dim
    )
    exact_attention_tensor = jnp.einsum('...LHD, ...THD->...LTH', query, key)
    exact_attention_tensor /= jnp.sqrt(qk_dim)
    exact_attention_tensor = jax.nn.softmax(exact_attention_tensor, axis=-2)
    exact_result = jnp.einsum(
        '...LTH, ...THD->...LHD', exact_attention_tensor, value
    )
    query_prime = sat_rfs(query, projection_matrix, is_query=True)
    key_prime = sat_rfs(key, projection_matrix, is_query=False)
    kv_tensor = jnp.einsum('...LHM, ...LHD->...HMD', key_prime, value)
    approx_result = jnp.einsum(
        '...LHM, ...HMD->...LHD', query_prime, kv_tensor
    )

    max_error = 1.2
    error = jnp.abs((exact_result - approx_result) / exact_result)
    self.assertLess(jnp.max(jnp.abs(error)), max_error)

  def test_relu(self):
    query = jnp.array([[1, 2], [3, 4], [1, -1], [1, 3]])
    key = jnp.array([[-3, 1], [1, 5], [-2, -3], [4, 1]])
    value = jnp.array([[-1], [2], [3], [1]])
    query = jnp.expand_dims(query, axis=-2)
    query = jnp.expand_dims(query, axis=0)
    key = jnp.expand_dims(key, axis=-2)
    key = jnp.expand_dims(key, axis=0)
    value = jnp.expand_dims(value, axis=-2)
    value = jnp.expand_dims(value, axis=0)
    query_prime = sat.general_kernel_linearization(query)
    key_prime = sat.general_kernel_linearization(key)
    kv = jnp.einsum('...lhm,...lhd->...hmd', key_prime, value)
    numerator = jnp.einsum('...lhm,...hmd->...lhd', query_prime, kv)
    key_prime_sum = jnp.sum(key_prime, axis=-3)
    denominator = jnp.einsum('...lhm,...hm->...lh', query_prime, key_prime_sum)
    denominator = jnp.expand_dims(
        denominator, len(denominator.shape)
    )
    result = numerator / denominator
    groundtruth = jnp.array([1.368421052, 1.34883720, 1.2, 1.3846153])
    groundtruth = jnp.expand_dims(groundtruth, axis=[0, -1, -2])
    max_error = 0.001
    error = jnp.abs(groundtruth - result)
    self.assertLess(jnp.max(jnp.abs(error)), max_error)


if __name__ == '__main__':
  absltest.main()
