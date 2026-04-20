"""Configuration generators for gvt."""

import jax.numpy as jnp
import tensorflow as tf


def get_tf_dtype(config):
  if config.dtype == "bfloat16":
    return tf.bfloat16
  return tf.float32


def get_jnp_dtype(config):
  if config.dtype == "bfloat16":
    return jnp.bfloat16
  return jnp.float32
