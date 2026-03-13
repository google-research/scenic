"""VQCAE Model."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.robust_vit.gvt import basic_res_arc
from scenic.projects.robust_vit.gvt import enc_dec_arc
from scenic.projects.robust_vit.gvt import losses


def l2_normalize(x, axis=None, epsilon=1e-12):
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return jnp.multiply(x, x_inv_norm)


class VectorQuantizer(nn.Module):
  """Basic vector quantizer."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  @nn.compact
  def __call__(
      self, x, **kwargs
  ):
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        "codebook",
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    if self.config.vqvae.get("latent_normalize", False):
      x = l2_normalize(x, axis=-1)
      codebook = l2_normalize(codebook, axis=-1)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    encoding_indices = jnp.argmin(distances, axis=-1)
    encodings = jax.nn.one_hot(
        encoding_indices, codebook_size, dtype=self.dtype)
    quantized = self.quantize(encodings)
    result_dict = dict()
    if self.train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x))**2)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            -distances,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
      q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
      entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict = dict(
          quantizer_loss=loss,
          e_latent_loss=e_latent_loss,
          q_latent_loss=q_latent_loss,
          entropy_loss=entropy_loss)
      quantized = x + jax.lax.stop_gradient(quantized - x)

    result_dict.update({
        "encodings": encodings,
        "encoding_indices": encoding_indices,
        "raw": x,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)
    if self.config.vqvae.get("latent_normalize", False):
      codebook = l2_normalize(codebook, axis=-1)
    return jnp.dot(z, codebook)

  def get_codebook(self) -> jnp.ndarray:
    return jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    codebook = self.variables["params"]["codebook"]
    if self.config.vqvae.get("latent_normalize", False):
      codebook = l2_normalize(codebook, axis=-1)
    return jnp.take(codebook, ids, axis=0)


class VladVQ(nn.Module):
  """Vlad vector quantizer, use negative distance as logit."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, **kwargs):
    num_centroids = self.config.vqvae.num_centroids
    codebook_size = self.config.vqvae.codebook_size
    tau = self.config.vqvae.tau
    codebook = self.param(
        "codebook",
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    weights = jax.nn.softmax(-distances / tau, axis=-1)
    # top_weights and top_indices: [batch_size, *image_size, num_centroids]
    top_weights, top_indices = jax.lax.top_k(weights, num_centroids)
    top_weights = top_weights / jnp.sum(top_weights, axis=-1, keepdims=True)
    indices_matrix = jax.nn.one_hot(top_indices, codebook_size, axis=-1)
    encodings = jnp.einsum("b...h, b...hd->b...d", top_weights, indices_matrix)
    result_dict = {
        "encoding_indices": top_indices,
        "encoding_weights": top_weights,
        "encodings": encodings,
        "raw": x,
    }
    quantized = self.decode_ids((top_indices, top_weights))
    if self.train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x))**2)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            -distances,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
      q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
      entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict.update(
          dict(
              quantizer_loss=loss,
              e_latent_loss=e_latent_loss,
              q_latent_loss=q_latent_loss,
              entropy_loss=entropy_loss))
      quantized = x + jax.lax.stop_gradient(quantized - x)
    return quantized, result_dict

  def decode_ids(self, x) -> jnp.ndarray:
    # Assumes |weights| has been l1 regularized.
    indices, weights = x
    codebook = jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)
    return jnp.sum(
        weights[..., jnp.newaxis] * jnp.take(codebook, indices, axis=0),
        axis=-2)


class VladVQDirect(nn.Module):
  """Vlad vector quantizer, directly predict logit instead of distance."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, **kwargs):
    num_centroids = self.config.vqvae.num_centroids
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        "codebook",
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    logits = nn.Dense(codebook_size, dtype=self.dtype)(x)
    if self.config.vqvae.topk == "softmax_first":
      weights = jax.nn.softmax(logits, axis=-1)
      # top_weights and top_indices: [batch_size, *image_size, num_centroids]
      top_weights, top_indices = jax.lax.top_k(weights, num_centroids)
      top_weights = top_weights / jnp.sum(top_weights, axis=-1, keepdims=True)
    elif self.config.vqvae.topk == "softmax_last":
      top_logits, top_indices = jax.lax.top_k(logits, num_centroids)
      top_weights = jax.nn.softmax(top_logits, axis=-1)
    else:
      raise NotImplementedError
    indices_matrix = jax.nn.one_hot(top_indices, codebook_size, axis=-1)
    encodings = jnp.einsum("b...h, b...hd->b...d", top_weights, indices_matrix)
    result_dict = {
        "encoding_indices": top_indices,
        "encoding_weights": top_weights,
        "encodings": encodings,
        "raw": x,
    }
    quantized = self.decode_ids((top_indices, top_weights))
    if self.train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x))**2)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            logits,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
      q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
      entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict.update(
          dict(
              quantizer_loss=loss,
              e_latent_loss=e_latent_loss,
              q_latent_loss=q_latent_loss,
              entropy_loss=entropy_loss))
      quantized = x + jax.lax.stop_gradient(quantized - x)
    return quantized, result_dict

  def decode_ids(self, x) -> jnp.ndarray:
    # Assumes |weights| has been l1 regularized.
    indices, weights = x
    codebook = jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)
    return jnp.sum(
        weights[..., jnp.newaxis] * jnp.take(codebook, indices, axis=0),
        axis=-2)


class GumbelVQ(nn.Module):
  """Gumbel VQ."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, *, tau=1.0):
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        "codebook",
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="uniform"),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    result_dict = dict()
    encoding_indices = jnp.argmin(distances, axis=-1)
    if self.train:
      noise = jax.random.gumbel(
          self.make_rng("rng"), distances.shape, dtype=self.dtype)
      encodings = jax.nn.softmax((-distances + noise) / tau, axis=-1)
      quantized = self.quantize(encodings)
    else:
      encodings = jax.nn.one_hot(
          encoding_indices, codebook_size, dtype=self.dtype)
      quantized = self.quantize(encodings)
    result_dict.update({
        "quantizer_loss": 0.0,
        "encodings": encodings,
        "encoding_indices": encoding_indices,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = jnp.asarray(
        self.variables["params"]["codebook"], dtype=self.dtype)
    return jnp.dot(z, codebook)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(self.variables["params"]["codebook"], ids, axis=0)


class VQVAE(nn.Module):
  """VQ-VAE model."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu

  def setup(self):
    """VQ-VAE setup."""
    if self.config.vqvae.quantizer == "gumbel":
      self.quantizer = GumbelVQ(
          config=self.config, train=self.train, dtype=self.dtype)
    elif self.config.vqvae.quantizer == "vq":
      self.quantizer = VectorQuantizer(
          config=self.config, train=self.train, dtype=self.dtype)
    elif self.config.vqvae.quantizer == "vlad":
      self.quantizer = VladVQ(
          config=self.config, train=self.train, dtype=self.dtype)
    elif self.config.vqvae.quantizer == "vlad_direct":
      self.quantizer = VladVQDirect(
          config=self.config, train=self.train, dtype=self.dtype)
    else:
      raise NotImplementedError
    self.logit_laplace_loss_ratio = self.config.vqvae.get(
        "logit_laplace_loss_ratio", 0.0)
    if self.logit_laplace_loss_ratio != 0.0:
      output_dim = 6
      self.use_logit_laplace = True
    else:
      output_dim = 3
      self.use_logit_laplace = False
    if self.config.vqvae.architecture == "basic_res_arc":
      self.encoder = basic_res_arc.EncoderResnetStack(
          config=self.config, train=self.train, dtype=self.dtype)
      self.decoder = basic_res_arc.DecoderResnetStack(
          config=self.config, train=self.train, dtype=self.dtype)
    elif self.config.vqvae.architecture == "enc_dec_arc":
      self.encoder = enc_dec_arc.Encoder(
          config=self.config, train=self.train, dtype=self.dtype)
      self.decoder = enc_dec_arc.Decoder(
          config=self.config,
          train=self.train,
          output_dim=output_dim,
          dtype=self.dtype)
    else:
      raise NotImplementedError

  def encode(self, input_dict):
    image = input_dict["image"]
    encoded_feature = self.encoder(image)
    if self.config.vqvae.quantizer == "gumbel" and self.train:
      quantized, result_dict = self.quantizer(
          encoded_feature, tau=input_dict["tau"])
    else:
      quantized, result_dict = self.quantizer(encoded_feature)
    return quantized, result_dict

  def decode(self, x: jnp.ndarray) -> jnp.ndarray:
    if not self.use_logit_laplace:
      reconstructed = self.decoder(x)
    else:
      decoder_out = self.decoder(x)
      decoded_mu = decoder_out[..., :3]
      reconstructed = jax.nn.sigmoid(decoded_mu)
      reconstructed = losses.log_laplace_postprocess(reconstructed)
    return reconstructed

  def get_codebook_funct(self):
    # This function only works for the naive VQGAN
    return self.quantizer.get_codebook()

  def decode_from_indices(self, inputs):
    if isinstance(inputs, dict):
      ids = inputs["encoding_indices"]
      if "vlad" in self.config.vqvae.quantizer:
        ids = (ids, inputs["encoding_weights"])
    else:
      ids = inputs
    features = self.quantizer.decode_ids(ids)
    reconstructed_image = self.decode(features)
    return reconstructed_image

  def encode_to_indices(self, inputs):
    if isinstance(inputs, dict):
      image = inputs["image"]
    else:
      image = inputs
    encoded_feature = self.encoder(image)
    _, result_dict = self.quantizer(encoded_feature)
    ids = result_dict["encoding_indices"]
    return ids

  def __call__(self, input_dict):
    quantized, result_dict = self.encode(input_dict)
    if not self.use_logit_laplace:
      outputs = self.decoder(quantized)
      result_dict["logit_laplace_loss"] = 0.0
    else:
      log_laplace_inputs = losses.log_laplace_preprocess(input_dict["image"])
      decoder_out = self.decoder(quantized)
      decoded_mu = decoder_out[..., :3]
      decoded_log_sigma = decoder_out[..., 3:]
      reconstructed = jax.nn.sigmoid(decoded_mu)
      outputs = losses.log_laplace_postprocess(reconstructed)
      # To be consistent with TF code, adding the constant
      _, log_laplace_loss = losses.log_laplace_loss(log_laplace_inputs,
                                                    decoded_mu,
                                                    decoded_log_sigma)
      log_laplace_loss = jnp.mean(log_laplace_loss)

      result_dict[
          "logit_laplace_loss"] = log_laplace_loss * self.logit_laplace_loss_ratio
    return outputs, result_dict
