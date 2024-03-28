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

"""Performers' attention library."""

# pylint: disable=invalid-name

import abc
import math
from typing import (Any, Dict, Optional, Sequence, Tuple, Union)

from flax.linen.linear import PrecisionLike

import jax
from jax import random
import jax.numpy as jnp

from scenic.projects.performer import subquadratic_attention as sat
from scenic.projects.performer import utils as ut

RANDOM_FEATURES_SEED = 873457891289
BIG_CONSTANT = 10000000.0
PERFORMERS_RPE_SEED = 73829861893
MAX_NB_PACKED_SEQS = 7

NUM_FT_PARAMS_PER_HEAD = 25
NUM_FT_RAND_FEATURES = 64

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class RandomMatrix(abc.ABC):
  r"""Abstract class providing a method for constructing 2D random arrays.

  Class is responsible for constructing 2D random arrays.
  """

  @abc.abstractmethod
  def get_2d_array(self):
    raise NotImplementedError('Abstract method')


class GaussianUnstructuredRandomMatrix(RandomMatrix):

  def __init__(self, nb_rows, nb_columns, key):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns
    self.key = random.PRNGKey(key)

  def get_2d_array(self):
    return random.normal(self.key, (self.nb_rows, self.nb_columns))


class GaussianOrthogonalRandomMatrix(RandomMatrix):
  r"""Class providing a method to create Gaussian orthogonal matrix.

  Class is responsible for constructing 2D Gaussian orthogonal arrays.
  """

  def __init__(self, nb_rows, nb_columns, key, scaling=0):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns

    rng = random.PRNGKey(key)
    matrixrng, _ = random.split(rng)

    self.key = matrixrng
    self.scaling = scaling

  def get_2d_array(self):
    nb_full_blocks = int(self.nb_rows / self.nb_columns)
    block_list = []
    rng = self.key
    for _ in range(nb_full_blocks):
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q)
    remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
    if remaining_rows > 0:
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q[:remaining_rows])
    final_matrix = jnp.vstack(block_list)

    if self.scaling == 0:
      rng, rng_input = jax.random.split(rng)
      multiplier = jnp.linalg.norm(
          random.normal(rng_input, (self.nb_rows, self.nb_columns)), axis=1)
    elif self.scaling == 1:
      multiplier = jnp.sqrt(float(self.nb_columns)) * jnp.ones((self.nb_rows))
    else:
      raise ValueError('Scaling must be one of {0, 1}. Was %s' % self.scaling)

    return jnp.matmul(jnp.diag(multiplier), final_matrix)


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR+ noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    vs: value tensor of the shape [B...,L,H,D].

  Returns:
    Not-normalized FAVOR+ noncausal attention AV.
  """
  kvs = jnp.einsum('...lhm,...lhd->...hmd', ks, vs)
  return jnp.einsum('...lhm,...hmd->...lhd', qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR+ normalizer in noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.

  Returns:
    FAVOR+ normalizer in noncausal attention.
  """
  ks_sum = jnp.sum(ks, axis=-3)
  return jnp.einsum('...lhm,...hm->...lh', qs, ks_sum)


def masked_numerator(qs, ks, vs, masker, mask):
  """Computes not-normalized FAVOR+ noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    vs: value tensor of the shape [B...,L,H,D].
    masker: object of the type masks.Mask applying masking mechanism using given
      mask.
    mask: compact encoding of the masking mechanism.

  Returns:
    Not-normalized masked FAVOR+ attention.
  """
  # See: Alg. 1 from https://arxiv.org/pdf/2107.07999.pdf.
  f1_tensor = jnp.reshape(
      jnp.einsum('...m,...d->...md', ks, vs),
      (ks.shape[0], ks.shape[1], ks.shape[2], ks.shape[-1] * vs.shape[-1]))
  d1_tensor = masker.act(mask, f1_tensor)
  d1_tensor_unflattened = jnp.reshape(
      d1_tensor, (d1_tensor.shape[0], d1_tensor.shape[1], d1_tensor.shape[2],
                  ks.shape[-1], vs.shape[-1]))
  return jnp.einsum('...m,...md->...d', qs, d1_tensor_unflattened)


def masked_denominator(qs, ks, masker, mask):
  """Computes masked FAVOR+ normalizer.

  Args:
    qs: query_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    ks: key_prime tensor of the shape [B...,L,H,M], where M stands for the
      number of kernel features.
    masker: object of the type masks.Mask applying masking mechanism using given
      mask.
    mask: compact encoding of the masking mechanism.

  Returns:
    FAVOR+ normalizer in masked FAVOR+ attention.
  """
  d2_tensor = masker.act(mask, ks)
  return jnp.einsum('...m,...m->...', qs, d2_tensor)


def generic_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.001,
                                  normalize_data=True,
                                  numerator_denominator_stabilizer=True,
                                  activation_fn=jax.nn.relu):
  r"""Computes features based on an activation (e.g.

  ReLU-kernel by default).

  By default, computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.
    activation_fn: activation function to use for the kernel transformation.
      Defaults to relu.

  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  del normalize_data
  del numerator_denominator_stabilizer
  if projection_matrix is None:
    return activation_fn(data) + numerical_stabilizer
  else:
    ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    data_dash = ratio * jnp.einsum('...lhd,md->...lhm', data, projection_matrix)
    kernel_feature_map = activation_fn(data_dash) + numerical_stabilizer
    return kernel_feature_map


def exp_softmax_kernel_transformation(data,
                                      is_query,
                                      projection_matrix=None,
                                      numerical_stabilizer=0.000001,
                                      normalize_data=True,
                                      numerator_denominator_stabilizer=True):
  r"""Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.

  Returns:
    Corresponding kernel feature map.
  """

  if projection_matrix is None:
    raise ValueError('projection_matrix cannot be unspecified for softmax '
                     'kernel.')
  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=data.ndim - 1, keepdims=True)
    lengths = jnp.sqrt(lengths)
    data /= lengths
    data *= jnp.sqrt(jnp.sqrt(data.shape[-1]))

  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_dash = jnp.einsum('...lhd,md->...lhm', data_normalizer * data,
                         projection_matrix)
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  if numerator_denominator_stabilizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      stab = jnp.max(data_dash, axis=last_dims_t, keepdims=True)
    else:
      stab = jnp.max(data_dash, keepdims=True)
    data_dash = ratio * (
        jnp.exp(data_dash - stab - diag_data) + numerical_stabilizer)
  else:
    data_dash = ratio * (jnp.exp(data_dash) + numerical_stabilizer)

  return data_dash


def expplus_softmax_kernel_transformation(
    base_data,
    extra_data,
    is_query,
    projection_matrix=None,
    numerical_stabilizer=0.000001,
    normalize_data=True,
    numerator_denominator_stabilizer=True):
  r"""Computes random features for the softmax kernel using FAVOR++ mechanism.

  Computes random features for the softmax kernel using FAVOR++ mechanism.

  Args:
    base_data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    extra_data: auxiliary data tensor of the. same shape as <base_data> for
      computing additional statistics to optimize the coefficients of the random
      maps.
    is_query: indicates whether input data is a query or key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
    normalize_data: whether queries/keys should \sqrt{d}-normalized.
    numerator_denominator_stabilizer: whether numerator and denominator in the
      normalized attention computation should be numerically stabilized.

  Returns:
    Corresponding kernel feature map.
  """
  data = base_data
  if normalize_data:
    data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
  else:
    data_normalizer = 1.0
    lengths = jnp.square(data)
    lengths = jnp.sum(lengths, axis=data.ndim - 1, keepdims=True)
    lengths = jnp.sqrt(lengths)
    data /= lengths
    data *= jnp.sqrt(jnp.sqrt(data.shape[-1]))

  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_dash = jnp.einsum('blhd,md->blhm', data_normalizer * data,
                         projection_matrix)
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)

  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  _, l, _, _ = base_data.shape

  first_sum_of_squares = jnp.square(data)
  first_sum_of_squares = jnp.sum(
      first_sum_of_squares, axis=(1, -1), keepdims=True)
  first_sum_of_squares *= (data_normalizer * data_normalizer)
  first_sum_of_squares /= l
  second_sum_of_squares = jnp.square(extra_data)
  second_sum_of_squares = jnp.sum(
      second_sum_of_squares, axis=(1, -1), keepdims=True)
  second_sum_of_squares *= (data_normalizer * data_normalizer)
  second_sum_of_squares /= l
  data_sum = jnp.sum(data, axis=(1,), keepdims=True)
  extra_data_sum = jnp.sum(extra_data, axis=(1,), keepdims=True)
  d_prod = jnp.einsum('blhd,blhd->blh', data_sum, extra_data_sum)
  d_prod = jnp.expand_dims(d_prod, axis=-1)
  d_prod *= (data_normalizer * data_normalizer)
  d_prod *= (2.0 / (l * l))
  ave = first_sum_of_squares + second_sum_of_squares + d_prod
  dim = projection_matrix.shape[-1]
  A = (1.0 / (4.0 * ave)) * (
      jnp.sqrt((2.0 * ave + dim) *
               (2.0 * ave + dim) + 8.0 * dim * ave) - 2.0 * ave - dim)
  A = (1.0 - 1.0 / A) / 8.0
  B = jnp.sqrt(1.0 - 4.0 * A)
  D = jnp.power(1.0 - 4.0 * A, dim / 4.0)

  diag_omega = jnp.square(projection_matrix)
  diag_omega = jnp.sum(diag_omega, axis=projection_matrix.ndim - 1)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = jnp.expand_dims(diag_omega, axis=0)
  diag_omega = A * diag_omega

  if numerator_denominator_stabilizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      stab = B * jnp.max(data_dash, axis=last_dims_t, keepdims=True)
    else:
      stab = B * jnp.max(data_dash, keepdims=True)
    data_dash = ratio * D * (
        jnp.exp(B * data_dash - stab - diag_data + diag_omega) +
        numerical_stabilizer)
  else:
    data_dash = ratio * D * (
        jnp.exp(B * data_dash - diag_data + diag_omega) + numerical_stabilizer)

  return data_dash


#------------------------------------------------------------------------------
# Performers-compatible Relative Positional Encoding mechanism.
#
# The implementation is taken from the following paper: "Relative Positional
# Encoding for Transformers with Linear Complexity"
# (github code: https://cifkao.github.io/spe/)
#------------------------------------------------------------------------------


def sinespe(rng_key,
            key_shape,
            num_realizations: int = 64,
            num_sines: int = 10):
  """Sinusoidal stochastic positional encoding.

  Args:
    rng_key: A PRNGKey.
    key_shape: The shape of keys and queries.
    num_realizations: The number of realizations of the stochastic process (R).
    num_sines: The number of sin and cos components (K).

  Returns:
    sinusoidal encoding.
  """
  length = key_shape[1]
  in_features = key_shape[-1]
  num_heads = key_shape[-2]
  params_shape = (num_heads, in_features, num_sines)
  functor = lambda *args: jax.random.normal(*args) - 4.
  freqs = functor(rng_key, params_shape)
  offsets = jax.random.normal(rng_key, params_shape)

  def init_gains(rng_key, shape):
    gains = jax.random.normal(rng_key, shape)
    return gains / (
        jnp.sqrt(jnp.linalg.norm(gains, axis=-1, keepdims=True)) / 2)

  gains = init_gains(rng_key, params_shape)

  # build omega_q and omega_k,
  # with shape (num_heads, keys_dim, length, 2*num_sines)
  indices = jnp.linspace(0, length - 1, length)

  # making sure the frequencies are in [0, 0.5]
  freqs = jax.nn.sigmoid(freqs[:, :, None, :]) / 2.

  phases_q = (2 * math.pi * freqs * indices[None, None, :, None] +
              offsets[:, :, None, :])
  omega_q = jnp.stack([jnp.cos(phases_q), jnp.sin(phases_q)],
                      axis=-1).reshape(num_heads, in_features, length,
                                       2 * num_sines)

  phases_k = (2 * math.pi * freqs * indices[None, None, :, None])
  omega_k = jnp.stack([jnp.cos(phases_k), jnp.sin(phases_k)],
                      axis=-1).reshape(num_heads, in_features, length,
                                       2 * num_sines)

  # gains is (num_heads, keys_dim, num_sines). Making them softplus-nonnegat.
  gains = jax.nn.softplus(gains)

  # now upsample it to 2 * num_sines
  gains = jnp.stack([gains, gains], axis=-1).reshape(num_heads, in_features,
                                                     2 * num_sines)

  # draw noise of appropriate shape
  z = jax.random.normal(
      rng_key,
      (1, num_heads, in_features, 2 * num_sines, num_realizations),
  ) / jnp.sqrt(num_sines * 2)

  # scale each of the 2*num_sines by the appropriate gain
  # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
  z = z * gains[None, ..., None]

  # computing the sum over the sines.
  # gets (1, num_heads, keys_dim, length, num_realizations)
  qbar = jnp.matmul(omega_q[None], z)
  kbar = jnp.matmul(omega_k[None], z)

  # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
  qbar = jnp.transpose(qbar, (0, 3, 1, 2, 4))
  kbar = jnp.transpose(kbar, (0, 3, 1, 2, 4))

  scale = jnp.sqrt(jnp.sqrt(jnp.reciprocal(num_realizations * in_features)))
  return scale * qbar, scale * kbar


def spegate(rng_key, spe_code):
  """Stochastic Positional Encoding gating mechanism.

  Args:
    rng_key: A PRNGKey.
    spe_code: the code of the stochastic positional encoding mechanism.

  Returns:
    qbar and kbar positional encodings.
  """
  qbar, kbar = spe_code

  ### gate = self.param('gate', kbar.shape[-3:-1], jax.random.normal)
  gate = jax.random.normal(rng_key, kbar.shape[-3:-1])

  # incorporate the constant bias for Pd if required. First draw noise
  # such that noise noise^T = 1, for each head, feature, realization.
  in_features = kbar.shape[-2]
  num_realizations = kbar.shape[-1]
  noise = jax.random.normal(rng_key, kbar.shape[-3:])
  noise = noise / jnp.sqrt(jnp.sqrt(in_features * num_realizations))
  # constrain the gate parameter to be in [0 1]
  gate = jax.nn.sigmoid(gate[..., None])
  # add to queries and keys.
  pe_coef, noise_coef = jnp.sqrt(gate), jnp.sqrt(1. - gate)
  qbar = pe_coef * qbar + noise_coef * noise
  kbar = pe_coef * kbar + noise_coef * noise

  return qbar, kbar


def apply_spe(keys, spe):
  # sum over the keys_dim after multiplying by queries and keys
  # spe is (1, max_len, ...), truncating and broadcasting over the batch
  return (spe[:, :keys.shape[1]] * keys[..., None]).sum(axis=-2)


############################# MASKED PERFORMER #################################
class Mask(abc.ABC):
  """API for the scalable attention masking mechanism.

  The API for the masking mechanism used to efficiently modulate attention with
  no explicit materialization of the attention matrix.
  """

  @abc.abstractmethod
  def act(self, mask: Sequence[jnp.ndarray],
          input_tensor: jnp.ndarray) -> jnp.ndarray:
    """Multiplies the stack of H masks M (shape [L, L] each) by the inp.

    tensor.

    We denote by L the length of the input sequence and by H the number of
    heads). Each mask of the stack is element-wise multiplied with the regular
    attention matrix in the brute-force masked attention model.

    The method implements the algorithm of multiplying each matrix M of the
    stack by a given input tensor of the shape [B..., L,H,F]. F stands for the
    feature/embedding dimension. The resulting tensor is of the shape
    [B..., L,H,F]. The stack of the masks is encoded by <mask>.
    The slice corresponding to fixed batch indices (B...) and a head index (H)
    of the resulting tensor is obtained my multiplying corresponding mask M
    with the matrix given by the corresponding slice of the input tensor
    (of shape [L, H]) (standard matrix-matrix multiplication, not element-wise).
    The masks M are usually not explicitly materialized to avoid quadratic in L
    time complexity, but are instead encoded in a compact way.

    Args:
      mask: a compact encoding of the masking mechanism.
      input_tensor: <float>[batch_dims, length, head_dims, emb_dim] array.

    Returns:
      <float>[batch_dims, length, head_dims, emb_dim] result of the
        multiplication.
    """
    raise NotImplementedError


class RPEMask(Mask):
  # TODO(kchoro): support a variant with the first CLS token which is 'special'
  # in a sense that its weight is always constant (e.g. 1) regardless of the
  # relative position.
  """Relative Positional Encoding masking mechanism.

  Relative Positional Encoding masking mechanism for which the corresponding
  mask is Toeplitz (not necessarily symmetric).

  The use_fft knob chooses between two implementations that return identical
  results up to numerical errors. For highest speed set use_fft to True on GPU,
  and False on TPU as jax.fft() is relatively slower compared to matrix
  multiplication on TPUs.
  TODO(stamas, kchoro): Improve efficiency further on TPU for small batch sizes
  (constructing the Toeplitz matrices is the bottleneck) and for very long
  sequences with >=8K tokens.
  """

  def __init__(self, use_fft: bool = True):
    self._act_method = self._act_fft if use_fft else self._act_einsum

  def _act_fft(self, exp_first_rpe_array: jnp.ndarray,
               exp_second_rpe_array: jnp.ndarray,
               input_tensor: jnp.ndarray) -> jnp.ndarray:
    """Computes the action of the Toeplitz matrix using FFT."""
    # <exp_rpe_params> encodes the circulaw rows of the circulant embeddings
    # of the Toeplitz matrices corresponding to the RPE mechanism. It is of the
    # shape [H, 2L] (different RPE mechanisms for different heads).
    exp_rpe_params = jnp.concatenate([
        exp_first_rpe_array,
        jnp.zeros(shape=(exp_first_rpe_array.shape[0], 1)), exp_second_rpe_array
    ],
                                     axis=1)
    # The method conducts fast Toeplitz matrix-matrix multiplication by
    # (see:  https://math.mit.edu/icg/resources/teaching/18.085-spring2015/
    # toeplitz.pdf):
    # (1) embedding (conceptually) Toeplitz matrix in the 2x larger circulant
    #     matrix,
    # (2) decomposing (conceptually) this larger circulant matrix C as:
    #     C = DFT * diag (DFT * c) * DFT^-1, where: DFT is the discrete Fourier
    #     transform matrix, c is the circulant row-vector defining C and DFT^-1
    #     is an inverse of DFT.
    # (3) left-multiplying <input_tensor> by DFT^-1 using Fast Fourier Transform
    #     FFT, computing diag (DFT * c) using FFT and finally: computing the
    #     Hadamard product with diag (DFT * c) and applying last time FFT.
    # (4) taking the part of the obtained tensor corresponding to the Toeplit
    #     submatrix of the circulant matrix C.
    #
    # The shape of the input and output tensor is [B, L, H, F], where: B - batch
    # dimension, L attention dimension, H - heads dimension and F - feature/
    # embeddings dimension.
    circ_vec_len = exp_rpe_params.shape[-1]
    diag_array = jnp.fft.fft(exp_rpe_params)
    inv_dft_trans = jnp.fft.ifft(input_tensor, n=circ_vec_len, axis=-3)
    had_product = jnp.einsum('...lhf,hl->...lhf', inv_dft_trans, diag_array)
    return jnp.real(
        jnp.fft.fft(had_product, n=circ_vec_len,
                    axis=-3)[:, 0:(exp_rpe_params.shape[-1] // 2), :, :])

  def _act_einsum(self, exp_first_rpe_array: jnp.ndarray,
                  exp_second_rpe_array: jnp.ndarray,
                  input_tensor: jnp.ndarray) -> jnp.ndarray:
    """Constructs the Toeplitz matrix explicitly and uses einsum."""

    # blakehechtman@'s recursive roll method from
    # https://github.com/google/jax/issues/1646#issuecomment-1139044324
    # modified to work with multiple heads (matrices) at once.
    #
    # This is the fastest on TPU of all the alternatives by far. It's slightly
    # slower on GPU than the best GPU friendly method based on reshaping.
    # However performance on GPU is less important as FFT is even faster there.
    #
    # Shape of x is [H, 2*L-1] on first call, returns shape [H, L, L]
    def toeplitz(x):
      if len(x.shape) == 2:
        x = jnp.expand_dims(x, axis=-1)  # shape [H, L, 1]
      # Keep appending rotated columns until we have enough.
      num_rows = x.shape[-2]
      num_cols = x.shape[-1]
      size_needed = num_rows // 2 + 1  # (==L)
      if num_cols >= size_needed:
        return x[:, :size_needed, :size_needed]
      r = jnp.roll(x, num_cols, axis=-2)
      return toeplitz(jnp.concatenate([x, r], axis=-1))

    rpe_matrices = toeplitz(
        jnp.concatenate([exp_first_rpe_array, exp_second_rpe_array], axis=1))
    # Matrix multiplication, j is the length-index we sum over, h is head-index,
    # f is embedding-index. l-th column of the RPE matrix has the dist(.,l)
    # values used for computing l-th token.
    return jnp.einsum('...jhf,hjl->...lhf', input_tensor, rpe_matrices)

  def act(self, mask: Sequence[jnp.ndarray],
          input_tensor: jnp.ndarray) -> jnp.ndarray:
    # The RPE masker is encoded with the two 2D arrays of shapes [H, L] and
    # [H, L - 1] respectively, where L stands for the length of the input
    # sequence and H for the number of heads. An ith row of the first array is
    # of the form: c^{i}_{1} = [b^{i}_{0,0},b^{i}_{0,1},...,b^{i}_{0,L-1}], and
    # the ith row of the second array is of the form: c^{i}_{2} =
    # [b^{i}_{L-1,0},...,b^{i}_{1,0}] where b^{i}_{i,j} encodes the relative
    # position distance between ith query and jth key in the ith head (the
    # b-entries that would be added to the corresponding logits entries in the
    # attention matrix in the brute-force masked attention mechanism).
    #
    # Note: We do not impose symmetry so the equality: b^{i}_{i,j} = b^{i}_{j,i}
    # does not necessarily need to hold.
    first_rpe_array, second_rpe_array = mask
    return self._act_method(
        jnp.exp(first_rpe_array), jnp.exp(second_rpe_array), input_tensor)


def favor_attention(query,
                    key,
                    value,
                    inputs_mask,
                    kernel_transformation,
                    num_features,
                    head_dim,
                    seed,
                    rpe_method=None,
                    num_realizations=64,
                    num_sines=10,
                    use_random_projections=True,
                    hybrid_global_size=0,
                    segment_ids: Optional[Array] = None,
                    data_dependent_kfs=False):
  """Computes bidirectional (noncausal) normalized FAVOR+ attention.

  Computes FAVOR+ linear attention from Performers,based on:
  "Rethinking Attention with Performers": https://arxiv.org/abs/2009.14794).
  The current variant is bidirectional (noncausal).

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    inputs_mask: <bool>[batch, length] array indicating True for non-padding
      tokens and False for padding.
    kernel_transformation: transformation used to get finite kernel features.
    num_features: number of kernel map features to be used.
    head_dim: query/key dimensionality (relevant only if RPE mechanism is turned
      off).
    seed: seed for constructing random features (relevant only for the
      approximate softmax method)
    rpe_method: relative positional encoding method to be used. If None then no
      RPE mechanism is used. Active RPE options currently include: 'sine'
      (trigonometric mechanism).
    num_realizations: number of samples for the stochastic RPE mechanism.
    num_sines: number of sin waves for the trigonometric RPE mechanism.
    use_random_projections: determines whether random or deterministic
      (canonical projections will be used).
    hybrid_global_size: If not zero use first hybrid_global_size tokens for
      global full attention.
    segment_ids: packing mask. The mask is the 2-dimensional tensor of the shape
      [B,L], where: B - batch didmension, L - length dimension. Each slice
      corresponding to the fixed index in the batch is of the form:
      [1,...1,2,...,2,...,N,...,N,0,...,0], where x...x corresponds to the
      tokens of a fixed sequence within the super-sequence of packed sequences,
      N is the total number of packed sequences in the slice and 0-tokens encode
      padding. Even though, we enumerate different sequences from left to right
      in the increasing order, the mechanism works for any enumeration.
    data_dependent_kfs: indicates whether computed random features use
      data-driven statistics.

  Returns:
    bidirectional normalized FAVOR+ attention.
  """
  projection_matrix = None
  # Apply positional encoding if this option is turned on:
  rng_key = jax.random.PRNGKey(PERFORMERS_RPE_SEED)
  if rpe_method:
    if rpe_method == 'sinespe':
      qbar, kbar = sinespe(
          rng_key,
          query.shape,
          num_sines=num_sines,
          num_realizations=num_realizations)
    else:
      raise NotImplementedError('Variant of the RPE not supported yet.')
    qbar, kbar = spegate(rng_key, (qbar, kbar))
    query = apply_spe(query, qbar)
    key = apply_spe(key, kbar)
    _, _, _, extended_head_dim = query.shape
    if use_random_projections:
      projection_matrix = GaussianOrthogonalRandomMatrix(
          num_features, extended_head_dim, seed).get_2d_array()
  else:
    if use_random_projections:
      projection_matrix = GaussianOrthogonalRandomMatrix(
          num_features, head_dim, seed).get_2d_array()
  if hybrid_global_size > 0:
    global_full_attn_output = full_attn(query[:, :hybrid_global_size, :, :],
                                        key, value, inputs_mask)
    query = query[:, hybrid_global_size:, :, :]
  if not data_dependent_kfs:
    query_prime = kernel_transformation(query, True, projection_matrix)
    key_prime = kernel_transformation(key, False, projection_matrix)
  else:
    query_prime = kernel_transformation(query, key, True, projection_matrix)
    key_prime = kernel_transformation(key, query, False, projection_matrix)

  # TODO(kchoro): Compare this variant with the one where
  # broadcast_to(new_shape) replaces reshaping + tiling.
  if segment_ids is None:
    if inputs_mask is not None:
      b, length, h, m = jnp.shape(key_prime)
      inputs_mask = jnp.tile(
          jnp.reshape(inputs_mask, [b, length, 1, 1]), [1, 1, h, m])
      key_prime = jnp.where(inputs_mask, key_prime, 0)
  else:
    # TODO(wgaj): Add a test if the segments_id goes above MAX_NB_PACKED_SEQS.
    b, length, h, m = jnp.shape(key_prime)
    # Introducing extra dimension so that padding can be re-interpreted as
    # multi-packing with different packing masks corresponding to different
    # sequences in the super-sequence.
    # TODO(kchoro): Compare this approach with the one not using the upper bound
    # on the number of packed sequences but rather for/while looping (the latter
    # will in all likelihood require custom gradient implementation for
    # efficiency).
    packing_mask = jnp.arange(1, MAX_NB_PACKED_SEQS + 1, 1)
    packing_mask = jnp.tile(
        jnp.reshape(packing_mask, [MAX_NB_PACKED_SEQS, 1, 1, 1, 1]),
        [1, b, length, h, m])
    segment_ids = jnp.tile(
        jnp.reshape(segment_ids, [1, b, length, 1, 1]),
        [MAX_NB_PACKED_SEQS, 1, 1, h, m])
    padded_inputs_mask = (segment_ids == packing_mask)
    key_prime = jnp.tile(
        jnp.reshape(key_prime, [1, b, length, h, m]),
        [MAX_NB_PACKED_SEQS, 1, 1, 1, 1])
    query_prime = jnp.tile(
        jnp.reshape(query_prime, [1, b, length, h, m]),
        [MAX_NB_PACKED_SEQS, 1, 1, 1, 1])
    key_prime = jnp.where(padded_inputs_mask, key_prime, 0)
    query_prime = jnp.where(padded_inputs_mask, query_prime, 0)

  av_attention = noncausal_numerator(query_prime, key_prime, value)
  attention_normalizer = noncausal_denominator(query_prime, key_prime)
  if segment_ids is not None:
    av_attention = jnp.sum(av_attention, axis=0, keepdims=False)
    attention_normalizer = jnp.sum(attention_normalizer, axis=0, keepdims=False)

  attention_normalizer = jnp.expand_dims(attention_normalizer,
                                         len(attention_normalizer.shape))
  attention_normalizer = jnp.where(attention_normalizer <= 0.0,
                                   jnp.ones(attention_normalizer.shape),
                                   attention_normalizer)

  attention_output = av_attention / attention_normalizer
  # TODO(kchoro): Add support for padding + hybrid.
  if hybrid_global_size > 0:
    attention_output = jnp.concatenate(
        [global_full_attn_output, attention_output], 1)
  return attention_output


def masked_favor_attention(query, key, value, masker, mask, kernel_config):
  """Computes masked FAVOR+ attention.

  Computes masked FAVOR+ linear attention for Performers,based on:
  "Rethinking Attention with Performers": https://arxiv.org/abs/2009.14794)
  and "From block-Toeplitz matrices to differential equations on graphs: towards
  a general theory for scalable masked Transformers":
  https://arxiv.org/abs/2107.07999.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    masker: object applying masking mechanism using given mask.
    mask: compact encoding of the masking mechanism.
    kernel_config: config dictionary defining kernel transformation to be used
      for attention.

  Returns:
    masked FAVOR+ attention.
  """
  projection_matrix = None

  if kernel_config['use_random_projections']:
    projection_matrix = GaussianOrthogonalRandomMatrix(
        kernel_config['num_features'], query.shape[-1],
        kernel_config['seed']).get_2d_array()

  if kernel_config['kernel_transformation'] == 'softmax':
    kernel_transformation = exp_softmax_kernel_transformation
  else:
    if kernel_config['kernel_transformation'] == 'relu':
      activation_fn = jax.nn.relu
    else:
      activation_fn = (lambda x: x * x * x * x)

    def gen_transformation(a, b, c):
      return generic_kernel_transformation(a, b, c, activation_fn=activation_fn)

    kernel_transformation = gen_transformation

  query_prime = kernel_transformation(query, True, projection_matrix)
  key_prime = kernel_transformation(key, False, projection_matrix)
  av_attention = masked_numerator(query_prime, key_prime, value, masker, mask)
  attention_normalizer = masked_denominator(query_prime, key_prime, masker,
                                            mask)
  attention_normalizer = jnp.expand_dims(attention_normalizer,
                                         len(attention_normalizer.shape))
  attention_output = av_attention / attention_normalizer

  return attention_output


def full_attn(query_matrix, key_matrix, value_matrix, attn_mask=None):
  """Applies kernel attention with query, key, value tensors.

  This function defines the computation inside `call` with projected
  multi-head Q, K, V inputs. Users can override this function for customized
  attention implementation.

  Args:
    query_matrix: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
    key_matrix: Projected key `Tensor` of shape `[B, S, N, key_dim]`.
    value_matrix: Projected value `Tensor` of shape `[B, S, N, value_dim]`.
    attn_mask: a boolean mask of shape `[B, S]`, that prevents attending to
      masked positions. Note that the mask is only appied to the keys. User may
      want to mask the output if query contains pads.

  Returns:
    attention_output: Multi-headed outputs of attention computation.
  """

  full_product = jnp.einsum('BFNH,BTNH->BFTN', query_matrix,
                            key_matrix)  # [B, F, T, N]
  if attn_mask is not None:
    attn_mask = attn_mask.astype(key_matrix.dtype)
    attn_mask = jnp.expand_dims(jnp.expand_dims(attn_mask, axis=1), axis=3)
    adder = (1.0 - attn_mask) * -10000.0
    full_product += adder
  full_product = jax.nn.softmax(full_product, axis=2)
  attention_output = jnp.einsum('BFTN,BTNO->BFNO', full_product,
                                value_matrix)  # [B, F, N, O]
  return attention_output


def compute_ft(toeplitz_params, points):
  """Computes a Fourier Transform in a given set of points.

  Computes the fourier transform (FT) in a give set of points. The FT is
  parameterized as a weighted sum of multi-dimensional Gaussian pdfs:
  tau(x) = sum_i w_i * exp(-(x-mu_i)**2 / sigma_i**2).

  Args:
    toeplitz_params: the parameters defining parameterization of the FT -
      weights w_i ceners mu_i and standard deviations sigma_i.
    points: points where the FT eneeds to be computed.

  Returns:
  """
  d = points.shape[-1]
  weights = toeplitz_params[:, :NUM_FT_PARAMS_PER_HEAD]
  mus = toeplitz_params[:, NUM_FT_PARAMS_PER_HEAD:((1 + d) *
                                                   NUM_FT_PARAMS_PER_HEAD)]
  mus = jnp.reshape(mus, (toeplitz_params.shape[0], NUM_FT_PARAMS_PER_HEAD, d))
  sqsigmas = toeplitz_params[:, ((1 + d) * NUM_FT_PARAMS_PER_HEAD):]
  sqsigmas = jnp.exp(sqsigmas)
  h = toeplitz_params.shape[0]
  b_points = jnp.broadcast_to(
      points, (NUM_FT_PARAMS_PER_HEAD, h, NUM_FT_RAND_FEATURES, d))
  b_points = jnp.transpose(b_points, [2, 1, 0, 3])
  b_points -= mus
  b_points = -b_points**2
  b_points = jnp.sum(b_points, axis=-1)
  b_points /= jnp.expand_dims(sqsigmas, axis=0)
  b_points = jnp.exp(b_points)
  b_points *= jnp.expand_dims(weights, axis=0)
  b_points = jnp.sum(b_points, axis=-1)
  return jnp.transpose(b_points, [1, 0])


def create_random_points(d, nb_rows, nb_columns, seed):
  return jax.random.normal(
      key=random.PRNGKey(seed), shape=(nb_rows, nb_columns, d))


def create_point_densities(points):
  squared_points = points * points / 2.0
  point_squared_lengths = jnp.sum(squared_points, axis=-1)
  return (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-point_squared_lengths)


def create_snippet(toeplitz_params, coords, coeff, length):
  """Function creating FLT snippet encoding RPE.

  Computes the fourier transform (FT) in a give set of points. The FT is
  parameterized as a weighted sum of multi-dimensional Gaussian pdfs:
  tau(x) = sum_i w_i * exp(-(x-mu_i)**2 / sigma_i**2).

  Args:
    toeplitz_params: the parameters defining parameterization of the FT -
      weights w_i ceners mu_i and standard deviations sigma_i.
    coords: coordinates encoding positions of the tokens.
    coeff: renormaliization coefficient.
    length: the number of all the tokens.

  Returns:
  """
  h, _ = toeplitz_params.shape
  d = coords.shape[-1]
  b = coords.shape[0]
  points = create_random_points(d, h, NUM_FT_RAND_FEATURES, 0)
  densities = create_point_densities(points)
  ft_matrix = compute_ft(toeplitz_params, points)
  ratios = ft_matrix / densities
  result = jnp.broadcast_to(coords, (h, b, length, d))
  result = jnp.einsum('hbld,hmd->bhlm', result, points)
  result = jnp.exp(2.0 * jnp.pi * 1j * coeff * result)
  result = jnp.einsum('bhlm,hm->blhm', result, ratios)
  return (1.0 / jnp.sqrt(NUM_FT_RAND_FEATURES)) * result


def sharp_masked_favor_attention(query,
                                 key,
                                 value,
                                 coords,
                                 toeplitz_params,
                                 inputs_mask,
                                 kernel_transformation,
                                 num_features,
                                 head_dim,
                                 seed,
                                 rpe_method=None,
                                 num_realizations=64,
                                 num_sines=10,
                                 use_random_projections=True,
                                 hybrid_global_size=0,
                                 segment_ids: Optional[Array] = None,
                                 data_dependent_kfs=False):
  """Computes linear attention supporting RPE masking through FLT.

  Computes linear attention supporting RPE through thr FLT mechanism, based on:
  "Learning a Fourier Transform for Linear Relative Positional Encodings in
   Transformers": https://arxiv.org/pdf/2302.01925.pdf.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    coords: coordinates of the tokens.
    toeplitz_params: leearnable parameters defining RPE mask.
    inputs_mask: <bool>[batch, length] array indicating True for non-padding
      tokens and False for padding.
    kernel_transformation: transformation used to get finite kernel features.
    num_features: number of kernel map features to be used.
    head_dim: query/key dimensionality (relevant only if RPE mechanism is turned
      off).
    seed: seed for constructing random features (relevant only for the
      approximate softmax method)
    rpe_method: relative positional encoding method to be used. If None then no
      RPE mechanism is used. Active RPE options currently include: 'sine'
      (trigonometric mechanism).
    num_realizations: number of samples for the stochastic RPE mechanism.
    num_sines: number of sin waves for the trigonometric RPE mechanism.
    use_random_projections: determines whether random or deterministic
      (canonical projections will be used).
    hybrid_global_size: If not zero use first hybrid_global_size tokens for
      global full attention.
    segment_ids: packing mask. The mask is the 2-dimensional tensor of the shape
      [B,L], where: B - batch didmension, L - length dimension. Each slice
      corresponding to the fixed index in the batch is of the form:
      [1,...1,2,...,2,...,N,...,N,0,...,0], where x...x corresponds to the
      tokens of a fixed sequence within the super-sequence of packed sequences,
      N is the total number of packed sequences in the slice and 0-tokens encode
      padding. Even though, we enumerate different sequences from left to right
      in the increasing order, the mechanism works for any enumeration.
    data_dependent_kfs: indicates whether computed random features use
      data-driven statistics.

  Returns:
    bidirectional normalized FAVOR+ attention spporting RPE through FLT.
  """
  projection_matrix = None
  # Apply positional encoding if this option is turned on:
  rng_key = jax.random.PRNGKey(PERFORMERS_RPE_SEED)
  if rpe_method:
    if rpe_method == 'sinespe':
      qbar, kbar = sinespe(
          rng_key,
          query.shape,
          num_sines=num_sines,
          num_realizations=num_realizations)
    else:
      raise NotImplementedError('Variant of the RPE not supported yet.')
    qbar, kbar = spegate(rng_key, (qbar, kbar))
    query = apply_spe(query, qbar)
    key = apply_spe(key, kbar)
    _, _, _, extended_head_dim = query.shape
    if use_random_projections:
      projection_matrix = GaussianOrthogonalRandomMatrix(
          num_features, extended_head_dim + 2 * NUM_FT_RAND_FEATURES,
          seed).get_2d_array()
  else:
    if use_random_projections:
      projection_matrix = GaussianOrthogonalRandomMatrix(
          num_features, head_dim + 2 * NUM_FT_RAND_FEATURES,
          seed).get_2d_array()
  if hybrid_global_size > 0:
    global_full_attn_output = full_attn(query[:, :hybrid_global_size, :, :],
                                        key, value, inputs_mask)
    query = query[:, hybrid_global_size:, :, :]

  b, l, h, d = jnp.shape(query)
  coe = jnp.sqrt(jnp.sqrt(d))
  q_snippet = create_snippet(toeplitz_params, coords, 1.0, l)
  q_snippet_imag = coe * jnp.imag(q_snippet)
  q_snippet_real = coe * jnp.real(q_snippet)
  k_snippet = create_snippet(toeplitz_params, coords, -1.0, l)
  k_snippet_imag = coe * jnp.imag(k_snippet)
  k_snippet_real = coe * jnp.real(k_snippet)

  new_query = jnp.concatenate([query, q_snippet_real, q_snippet_imag], axis=-1)
  new_key = jnp.concatenate([key, k_snippet_real, -k_snippet_imag], axis=-1)

  if not data_dependent_kfs:
    query_prime = kernel_transformation(new_query, True, projection_matrix)
    key_prime = kernel_transformation(new_key, False, projection_matrix)
  else:
    query_prime = kernel_transformation(new_query, key, True, projection_matrix)
    key_prime = kernel_transformation(new_key, query, False, projection_matrix)

  # TODO(kchoro): Compare this variant with the one where
  # broadcast_to(new_shape) replaces reshaping + tiling.
  if segment_ids is None:
    if inputs_mask is not None:
      b, length, h, m = jnp.shape(key_prime)
      inputs_mask = jnp.tile(
          jnp.reshape(inputs_mask, [b, length, 1, 1]), [1, 1, h, m])
      key_prime = jnp.where(inputs_mask, key_prime, 0)
  else:
    # TODO(wgaj): Add a test if the segments_id goes above MAX_NB_PACKED_SEQS.
    b, length, h, m = jnp.shape(key_prime)
    # Introducing extra dimension so that padding can be re-interpreted as
    # multi-packing with different packing masks corresponding to different
    # sequences in the super-sequence.
    # TODO(kchoro): Compare this approach with the one not using the upper bound
    # on the number of packed sequences but rather for/while looping (the latter
    # will in all likelihood require custom gradient implementation for
    # efficiency).
    packing_mask = jnp.arange(1, MAX_NB_PACKED_SEQS + 1, 1)
    packing_mask = jnp.tile(
        jnp.reshape(packing_mask, [MAX_NB_PACKED_SEQS, 1, 1, 1, 1]),
        [1, b, length, h, m])
    segment_ids = jnp.tile(
        jnp.reshape(segment_ids, [1, b, length, 1, 1]),
        [MAX_NB_PACKED_SEQS, 1, 1, h, m])
    padded_inputs_mask = (segment_ids == packing_mask)
    key_prime = jnp.tile(
        jnp.reshape(key_prime, [1, b, length, h, m]),
        [MAX_NB_PACKED_SEQS, 1, 1, 1, 1])
    query_prime = jnp.tile(
        jnp.reshape(query_prime, [1, b, length, h, m]),
        [MAX_NB_PACKED_SEQS, 1, 1, 1, 1])
    key_prime = jnp.where(padded_inputs_mask, key_prime, 0)
    query_prime = jnp.where(padded_inputs_mask, query_prime, 0)

  av_attention = noncausal_numerator(query_prime, key_prime, value)
  attention_normalizer = noncausal_denominator(query_prime, key_prime)
  if segment_ids is not None:
    av_attention = jnp.sum(av_attention, axis=0, keepdims=False)
    attention_normalizer = jnp.sum(attention_normalizer, axis=0, keepdims=False)

  attention_normalizer = jnp.expand_dims(attention_normalizer,
                                         len(attention_normalizer.shape))
  attention_normalizer = jnp.where(attention_normalizer <= 0.0,
                                   jnp.ones(attention_normalizer.shape),
                                   attention_normalizer)

  attention_output = av_attention / attention_normalizer
  # TODO(kchoro): Add support for padding + hybrid.
  if hybrid_global_size > 0:
    attention_output = jnp.concatenate(
        [global_full_attn_output, attention_output], 1)
  return attention_output


def regular_performer_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    toeplitz_params: Array = None,
    kernel_config: Union[Dict[Any, Any], None] = None):
  """Wrapper function for computing bidirectional normalized FAVOR+ attention.

  Wrapper function for computing FAVOR+ linear attention from Performers, based
  on: "Rethinking Attention with Performers": https://arxiv.org/abs/2009.14794).
  The current variant is bidirectional (noncausal).

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    bias: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch..., num_heads, q_length, kv_length]`
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
    mask: mask to be added to the attention matrix.
    broadcast_dropout: broadcast dropout indicator,
    dropout_rng: Optional JAX PRNGKey to be used for dropout.
    dropout_rate: float = dropout rate,
    deterministic: Deterministic or not (to apply dropout).
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    toeplitz_params: parameters defining the RPE mechanism,
    kernel_config: configuratiion of the attention kernel in Performers:

  Returns:
    bidirectional normalized FAVOR+ attention.
  """
  del bias
  del mask
  del broadcast_dropout
  del dropout_rng
  del dropout_rate
  del deterministic
  del dtype
  del precision
  del toeplitz_params

  if kernel_config['kernel_transformation'] == 'softmax':
    kernel_transformation = exp_softmax_kernel_transformation
  else:
    if kernel_config['kernel_transformation'] == 'relu':
      activation_fn = jax.nn.relu
    else:
      activation_fn = (lambda x: x * x * x * x)

    def gen_transformation(a, b, c):
      return generic_kernel_transformation(a, b, c, activation_fn=activation_fn)

    kernel_transformation = gen_transformation
  return favor_attention(
      query,
      key,
      value,
      inputs_mask=None,
      kernel_transformation=kernel_transformation,
      num_features=kernel_config['num_features'],
      head_dim=query.shape[-1],
      seed=0,
      rpe_method=kernel_config['rpe_method'],
      num_realizations=kernel_config['num_realizations'],
      num_sines=kernel_config['num_sines'],
      use_random_projections=kernel_config['use_random_projections'],
      hybrid_global_size=0,
      segment_ids=None,
      data_dependent_kfs=False)


def masked_performer_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    toeplitz_params: Array = None,
    kernel_config: Union[Dict[Any, Any], None] = None):
  """Wrapper function for computing linear attention supporting general RPE.

  Wrapper function for computing linear attention supporting general RPE,
  based on: "Stable, Fast and Accurate: Kernelized Attention with Relative
  Positional Encoding": https://arxiv.org/abs/2106.12566.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    bias: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch..., num_heads, q_length, kv_length]`
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
    mask: mask to be added to the attention matrix.
    broadcast_dropout: broadcast dropout indicator,
    dropout_rng: Optional JAX PRNGKey to be used for dropout.
    dropout_rate: float = dropout rate,
    deterministic: Deterministic or not (to apply dropout).
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    toeplitz_params: parameters defining the RPE mechanism,
    kernel_config: configuratiion of the attention kernel in Performers:

  Returns:
    bidirectional normalized FAVOR+ attention supporting general RPE.
  """
  del bias
  del mask
  del broadcast_dropout
  del dropout_rng
  del dropout_rate
  del deterministic
  del dtype
  del precision

  length = query.shape[-3]
  mask = (toeplitz_params[:, :length], toeplitz_params[:, length:])
  masker = RPEMask()
  return masked_favor_attention(query, key, value, masker, mask, kernel_config)


def sharp_masked_performer_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    coords: Array,  # [B, L, C]
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    toeplitz_params: Array = None,
    kernel_config: Union[Dict[Any, Any], None] = None):
  """Wrapper function for computing linear attention supporting RPE with FLT.

  Wrapper function for computing linear attention supporting RPE through FLT
  the mechanism, based on: "Learning a Fourier Transform for Linear Relative
  Positional Encodings in Transformers": https://arxiv.org/pdf/2302.01925.pdf.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    coords: coordinates of the tokens.
    bias: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch..., num_heads, q_length, kv_length]`
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
    mask: mask to be added to the attention matrix.
    broadcast_dropout: broadcast dropout indicator,
    dropout_rng: Optional JAX PRNGKey to be used for dropout.
    dropout_rate: float = dropout rate,
    deterministic: Deterministic or not (to apply dropout).
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    toeplitz_params: parameters defining the RPE mechanism,
    kernel_config: configuratiion of the attention kernel in Performers:

  Returns:
    bidirectional normalized FAVOR+ attention supporting RPE through FLT.
  """
  del bias
  del mask
  del broadcast_dropout
  del dropout_rng
  del dropout_rate
  del deterministic
  del dtype
  del precision
  if kernel_config['kernel_transformation'] == 'softmax':
    kernel_transformation = exp_softmax_kernel_transformation
  else:
    if kernel_config['kernel_transformation'] == 'relu':
      activation_fn = jax.nn.relu
    else:
      activation_fn = (lambda x: x * x * x * x)

    def gen_transformation(a, b, c):
      return generic_kernel_transformation(a, b, c, activation_fn=activation_fn)

    kernel_transformation = gen_transformation
  return sharp_masked_favor_attention(
      query,
      key,
      value,
      coords,
      toeplitz_params,
      inputs_mask=None,
      kernel_transformation=kernel_transformation,
      num_features=kernel_config['num_features'],
      head_dim=query.shape[-1],
      seed=0,
      rpe_method=kernel_config['rpe_method'],
      num_realizations=kernel_config['num_realizations'],
      num_sines=kernel_config['num_sines'],
      use_random_projections=kernel_config['use_random_projections'],
      hybrid_global_size=0,
      segment_ids=None,
      data_dependent_kfs=False)


def pseudolocal_subquadratic_attention(
    query: Array,   #  shape: [...M,H,D]
    key: Array,     #  shape: [...N,H,D]
    value: Array,   #  shape: [...N,H,D]
    coords: Array,  #  shape: [...M,E]   E=3
    aniso_matrix: Array,  # shape: [R,E]
    rf_type: Any,  #  'regular' | 'hyper'
    nb_rfs: int,
):
  """Pseudolocal Performer attention with Gaussian smoothing.

  Pseudolocal Performer attention with Gaussian smoothing

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    coords: coordinates of the tokens.
    aniso_matrix: matrix defining the anisotripicity of the Gaussian smoothing
    rf_type: type of the random feature mechanism applied ('regular' or 'hyper')
    nb_rfs: number of random features used

  Returns:
    bidirectional pseudolocal Performer's attention.
  """
  qk_dim = query.shape[-1]
  if rf_type == 'hyper':
    rfs = sat.softmax_hyper_positive_rfs
  else:
    rfs = sat.softmax_positive_rfs
  projection_matrix = ut.get_gaussian_orth_rand_mat(
      random.PRNGKey(0), nb_rfs, qk_dim + aniso_matrix.shape[0]
  )
  qk_normalizer = jnp.sqrt(jnp.sqrt(qk_dim))
  n_query = query / qk_normalizer
  n_key = key / qk_normalizer
  coords_proj = jnp.einsum('RE,...ME->...MR', aniso_matrix, coords)
  coords_proj = jnp.expand_dims(coords_proj, axis=-2)
  n_query_xyz = jnp.concatenate([n_query, coords_proj], axis=-1)
  n_key_xyz = jnp.concatenate([n_key, coords_proj], axis=-1)
  xyz_mult = jnp.exp(
      -0.5 * jnp.sum(jnp.square(coords_proj), axis=-1, keepdims=True)
  )
  n_query_xyz *= xyz_mult
  n_key_xyz *= xyz_mult
  query_prime = rfs(n_query_xyz, projection_matrix, is_query=True)
  key_prime = rfs(n_key_xyz, projection_matrix, is_query=False)
  kv = jnp.einsum('...lhm,...lhd->...hmd', key_prime, value)
  numerator = jnp.einsum('...lhm,...hmd->...lhd', query_prime, kv)
  key_prime_sum = jnp.sum(key_prime, axis=-3)
  denominator = jnp.einsum('...lhm,...hm->...lh', query_prime, key_prime_sum)
  denominator = jnp.expand_dims(
      denominator, len(denominator.shape)
  )
  result = numerator / denominator
  return result

