"""Discrete Vision Transformer that uses pretrained vq encoder's token as input."""

from typing import Any, Callable, Dict, Iterable, Optional

from absl import logging
import flax
import flax.linen as nn
from flax.linen.attention import dot_product_attention
from flax.linen.linear import DenseGeneral
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.robust_vit.gvt import vqvae
import scipy

Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape,
                    inputs.dtype)
    return inputs + pe


class ConcateatPositionEmbs(nn.Module):
  """Concatenate learned positional embeddings to the inputs.

  Attributes:
    hidden_size: the size for the transformer, which is our target size
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """
  hidden_size: int
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], self.hidden_size - inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape,
                    inputs.dtype)
    zeros = jnp.zeros((inputs.shape[0], 1, 1), inputs.dtype)
    pe = zeros + pe
    return jnp.concatenate([inputs, pe], 2)


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  add_pos: bool = False

  def get_stochastic_depth_mask(self, x: jnp.ndarray,
                                deterministic: bool) -> jnp.ndarray:
    """Generate the stochastic depth mask in order to apply layer-drop.

    Args:
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Stochastic depth mask.
    """
    if not deterministic and self.stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.stochastic_depth, shape)
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = x * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    return y * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + x


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  grid_height: int
  grid_width: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32
  add_pos: bool = False
  cat_pos: bool = False
  hidden_size: int = 0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    if self.cat_pos:
      x = ConcateatPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          hidden_size=self.hidden_size,
          name='posembed_input')(
              inputs)
    else:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              inputs)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder.
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
          self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
          add_pos=self.add_pos)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class VitBackbone(nn.Module):
  """Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  dtype: Any = jnp.float32
  is_fusion: bool = False
  in_downsample: int = 1
  add_pos: bool = False
  cat_pos: bool = False
  fusion_dim: int = 128
  gate: bool = False
  gate_dim: int = 64
  gate_latent: int = 32
  cross_att: bool = False
  f_head: int = 1
  cross_att_single: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               x_pixel: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False,
               ):
    """Apply VQViT on the inputs.

    Args:
      x: Input image. x now is the embedding from the vqgan, it's size is
      Bx14x14x256 if input image size is 224x224.
      x_pixel: raw pixel image.
      train: Indicator for is training.
      debug: Indicator for debugging.
    Returns:
      Output after VQViT.
    """
    _, w, h, c = x.shape
    if w == 28:  # if the block is not 14 by 14, but 28 by 28, then downsample.
      x = nn.Conv(
          c, (2, 2), strides=(2, 2), padding='VALID', name='embedding_down')(
              x)

    if self.patches.get('grid') is not None:
      gh, gw = self.patches.grid
    else:
      n, gh, gw, c = x.shape
      fh, fw = self.patches.size

    if not self.cat_pos:
      # Only concatenate input shortcut, or use linear projection to change
      # channel number when not catenate positional embedding
      if self.is_fusion:
        if self.hidden_size:
          n, gh, gw, c = x.shape
          fh, fw = self.patches.size
          if self.gate:
            gate_dim = self.gate_dim
            outdim = gate_dim if self.hidden_size - c > gate_dim else self.hidden_size - c - 1
            # SubNetWork for Gate Layer
            h1 = nn.Conv(
                outdim, (fh, fw),
                strides=(fh, fw),
                padding='VALID',
                name='gate_mlp')(
                    x_pixel)
            hc = jnp.concatenate([h1, x], axis=3)
            h2 = nn.Conv(
                self.gate_latent, (1, 1),
                strides=(1, 1),
                padding='VALID',
                name='gate_mlp2')(hc)
            h2 = nn.relu(h2)
            g = nn.Conv(
                1, (1, 1),
                strides=(1, 1),
                padding='SAME',
                name='gate_mlp3')(h2)
            g = jnp.reshape(g, (n, gh, gw))
            g = nn.softmax(g, axis=-1)
            g = jnp.reshape(g, (n, gh, gw, 1))

            # Pixel representation
            part_c = self.fusion_dim
            outdim = part_c if self.hidden_size - c > part_c else self.hidden_size - c - 1
            x_p = nn.Conv(
                outdim, (fh, fw),
                strides=(fh, fw),
                padding='VALID',
                name='embedding')(
                    x_pixel)
            x_p_g = x_p * g
            zeros = jax.numpy.zeros((n, gh, gw, self.hidden_size - c - outdim),
                                    dtype=x_p.dtype)
            x = jnp.concatenate([x_p_g, x, zeros], axis=3)

          elif self.cross_att:
            # multi-head attention
            # use linear projection for all query, key, and value
            # use pixel to attend discrete repre,
            # and use discrete representations to attend to pixel repre
            # merge both results together
            f_head = self.f_head
            query = nn.Conv(
                c*f_head, (fh, fw),
                strides=(fh, fw),
                padding='VALID',
                name='gate_mlp')(
                    x_pixel)
            p_value = nn.Conv(
                c*f_head, (fh, fw),
                strides=(fh, fw),
                padding='VALID',
                name='P_value_mlp')(
                    x_pixel)
            x_value = nn.Conv(
                c * f_head, (1, 1),
                strides=(1, 1),
                padding='VALID',
                name='x_value_mlp')(x)
            query = jnp.reshape(query, (n, gh*gw, f_head, c))
            x_value = jnp.reshape(x_value, (n, gh*gw, f_head, c))
            p_value = jnp.reshape(p_value, (n, gh*gw, f_head, c))
            out = dot_product_attention(query, x_value, x_value)
            out = DenseGeneral(
                features=c,
                axis=(-2, -1),
                name='fu_out')(
                    out)

            out2 = dot_product_attention(x_value, query, p_value)
            out2 = DenseGeneral(
                features=self.fusion_dim,
                axis=(-2, -1),
                name='fu_out2')(
                    out2)

            out = jnp.reshape(out, (n, gh, gw, c))
            out = out + x
            out2 = jnp.reshape(out2, (n, gh, gw, self.fusion_dim))
            assert c + self.fusion_dim <= self.hidden_size
            if c + self.fusion_dim == self.hidden_size:
              x = jnp.concatenate([out, out2], axis=3)
            else:
              zerodim = self.hidden_size - c - self.fusion_dim
              zeros = jax.numpy.zeros((n, gh, gw, zerodim), dtype=x_pixel.dtype)
              x = jnp.concatenate([out, out2, zeros], axis=3)

          elif self.cross_att_single:
            # use discrete representation as query to attend to pixel repre
            f_head = self.f_head
            key = nn.Conv(
                c*f_head, (fh, fw),
                strides=(fh, fw),
                padding='VALID',
                name='key_mlp')(
                    x_pixel)

            query = nn.Conv(
                c * f_head, (5, 5),
                strides=(1, 1),
                padding='SAME',
                name='query_mlp')(x)

            query = jnp.reshape(query, (n, gh*gw, f_head, c))
            key = jnp.reshape(key, (n, gh*gw, f_head, c))
            value = key
            out = dot_product_attention(query, key, value)
            out = DenseGeneral(
                features=self.fusion_dim,
                axis=(-2, -1),
                name='fu_out')(
                    out)
            out = jnp.reshape(out, (n, gh, gw, c))
            assert c + self.fusion_dim <= self.hidden_size
            if c + self.fusion_dim == self.hidden_size:
              x = jnp.concatenate([x, out], axis=3)
            else:
              zerodim = self.hidden_size - c - self.fusion_dim
              zeros = jax.numpy.zeros((n, gh, gw, zerodim), dtype=x_pixel.dtype)
              x = jnp.concatenate([x, out, zeros], axis=3)
          else:
            # Concatenate for Fusion, the algorithm that we finally use
            part_c = self.fusion_dim
            outdim = part_c if self.hidden_size - c > part_c else self.hidden_size - c - 1
            x_p = nn.Conv(
                outdim, (fh, fw),
                strides=(fh, fw),
                padding='VALID',
                name='embedding')(
                    x_pixel)
            zeros = jax.numpy.zeros((n, gh, gw, self.hidden_size - c - outdim),
                                    dtype=x_p.dtype)
            x = jnp.concatenate([x_p, x, zeros], axis=3)
      else:
        if self.hidden_size:
          # only used when dvae, or other mismatched resolution
          x = nn.Conv(
              self.hidden_size, (self.in_downsample, self.in_downsample),
              strides=(self.in_downsample, self.in_downsample),
              padding='VALID',
              name='linear_projection')(
                  x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='Transformer',
        grid_height=gh,
        grid_width=gw,
        add_pos=self.add_pos,
        cat_pos=self.cat_pos,
        hidden_size=self.hidden_size)(
            x, train=train)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x


class VQViT(nn.Module):
  """VQGAN embedding + ViT model.

  Attributes:
  config: configure
  dataset_meta_data: dataset information
  """
  config: ml_collections.ConfigDict
  dataset_meta_data: Dict[str, Any]

  def setup(self):
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    # if self.config.get('dvae', False):
    #   self.vqgan = encoder_lib.Encoder(dtype=model_dtype)
    # else:
    self.vqgan = vqvae.VQVAE(self.config, train=False, dtype=model_dtype)
    self.vit = VitBackbone(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        dtype=model_dtype,
        is_fusion=self.config.get('is_fusion', False),
        in_downsample=self.config.get('in_downsample', 1),
        add_pos=self.config.get('att_add_pos', False),
        cat_pos=self.config.get('cat_pos', False),
        fusion_dim=self.config.get('fusion_dim', 128),
        gate=self.config.get('gate', False),
        gate_dim=self.config.get('gate_dim', 64),
        gate_latent=self.config.get('gate_latent', 32),
        cross_att=self.config.get('cross_att', False),
        f_head=self.config.get('f_head', 1),
        cross_att_single=self.config.get('cross_att_single', False),
        )
    if self.config.get('dvae', False):
      codebook_size = 8192
      code_embedding_retrain = self.param(
          'code_embedding_retrain',
          jax.nn.initializers.variance_scaling(
              scale=1.0, mode='fan_in', distribution='uniform'),
          (codebook_size, self.config.model.hidden_size))
      self.code_embedding_retrain = jnp.asarray(code_embedding_retrain,
                                                dtype=model_dtype)
    elif self.config.get('retrain_embed_code', False):
      codebook_size = self.config.vqvae.codebook_size
      code_embedding_retrain = self.param(
          'code_embedding_retrain',
          jax.nn.initializers.variance_scaling(
              scale=1.0, mode='fan_in', distribution='uniform'),
          (codebook_size, self.config.vqvae.embedding_dim))
      self.code_embedding_retrain = jnp.asarray(code_embedding_retrain,
                                                dtype=model_dtype)
    elif self.config.get('finetune_embed_code', False):
      codebook_size = self.config.vqvae.codebook_size
      # Initialize our delta codebook for finetuning
      code_embedding_delta = self.param(
          'code_embedding_delta',
          jax.nn.initializers.zeros,
          (codebook_size, self.config.vqvae.embedding_dim))
      self.code_embedding_delta = jnp.asarray(code_embedding_delta,
                                              dtype=model_dtype)

  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False):
    # format the x as dict, and convert from range (-1,1) to (0,1)
    if self.config.get('input_gaussian', False) and train:
      x = x+ (
          jax.random.normal(self.make_rng('normal1'), shape=x.shape)*
          self.config.get('in_gaussian_std', 0) *
          random.uniform(self.make_rng('uniform')))
      x = jnp.clip(x, -1, 1)

    x_vq_input = {'image': (x + 1.0) / 2.0}

    if self.config.get('dvae', False):
      logits = self.vqgan(x_vq_input['image'])
      one_hot = jax.nn.one_hot(jnp.argmax(logits, -1), 8192, dtype=jnp.float32)
      one_hot = jax.lax.stop_gradient(one_hot)
      embedding = jnp.dot(one_hot, self.code_embedding_retrain)
      return self.vit(embedding,
                      x_pixel=x, train=train)
    else:
      quantized, embedding_dict = self.vqgan.encode(x_vq_input)
    if self.config.get('use_raw_vqencode', False):
      if not self.config.get('finetune_gan', False):
        embedding = jax.lax.stop_gradient(embedding_dict['raw'])
      else:
        embedding = embedding_dict['raw']
    elif self.config.get('retrain_embed_code', False):
      one_hot = embedding_dict['encodings']
      embedding = jnp.dot(one_hot, self.code_embedding_retrain)
      embedding = jax.lax.stop_gradient(embedding)
    elif self.config.get('finetune_embed_code', False):
      original_codebook = self.vqgan.get_codebook_funct()
      one_hot = jax.lax.stop_gradient(embedding_dict['encodings'])
      embedding = jnp.dot(one_hot, self.code_embedding_delta +
                          jax.lax.stop_gradient(original_codebook))
    elif self.config.get('both_vq_raw', False):
      original_codebook = self.vqgan.get_codebook_funct()
      one_hot = jax.lax.stop_gradient(embedding_dict['encodings'])
      ftvq_embedding = jnp.dot(
          one_hot,
          self.code_embedding_delta + jax.lax.stop_gradient(original_codebook))
      embedding = jnp.concatenate(
          [jax.lax.stop_gradient(embedding_dict['raw']), ftvq_embedding],
          axis=3)
    else:
      embedding = jax.lax.stop_gradient(quantized)
    if self.config.get('latent_gaussian', False) and train:
      embedding = embedding + (
          jax.random.normal(self.make_rng('normal2'), shape=quantized.shape) *
          (self.config.get('gaussian_std', 0) *
           random.uniform(self.make_rng('uniform2'))))

    return self.vit(embedding, x_pixel=x, train=train)


class RobViTMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Vision Transformer model for multi-label classification task."""

  def build_flax_model(self) -> nn.Module:

    return VQViT(config=self.config, dataset_meta_data=self.dataset_meta_data)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                representation_size=16,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                patches={'grid': (4, 4)},
                classifier='gap',
                data_dtype_str='float32')
    })

  def init_ganpart_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return init_vqgan_from_train_state(train_state, restored_train_state)

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return init_vit_from_train_state(train_state, restored_train_state,
                                     self.config, restored_model_cfg)


def init_vqgan_from_train_state(train_state: Any,
                                restored_train_state: Any) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is writen to load only vqgan from state. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    train_state: Any.
    restored_train_state: Any.

  Returns:
    Updated train_state.
  """
  params = flax.core.unfreeze(train_state.optimizer.target)
  restored_params = flax.core.unfreeze(
      restored_train_state['g_optimizer']['target'])
  params['vqgan'] = restored_params  # Maybe need a loop over items to assign.

  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))


def init_vit_from_train_state(
    train_state: Any, restored_train_state: Any,
    model_cfg: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is writen to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    model_cfg: Configuration of the model. Usually used for some asserts.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.

  Returns:
    Updated train_state.
  """
  params = flax.core.unfreeze(train_state.optimizer.target)
  restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)

  # Start moving parameters, one-by-one and apply changes if needed.
  for b_key, b_params in restored_params.items():
    if b_key == 'code_embedding_delta':
      params[b_key] = b_params
      continue
    for m_key, m_params in b_params.items():
      if m_key == 'output_projection':
        # For the classifier head, we use a the randomly initialized params and
        #   ignore the one from pretrained model.
        pass

      elif m_key == 'pre_logits':
        if model_cfg.model.representation_size is None:
          # We don't have representation_size in the new model, so let's ignore
          #   it from the pretained model, in case it has it.
          # Note, removing the key from the dictionary is necessary to prevent
          #   obscure errors from the Flax optimizer.
          params[b_key].pop(m_key, None)
        else:
          assert restored_model_cfg.model.representation_size
          params[b_key][m_key] = m_params

      elif m_key == 'Transformer':
        for tm_key, tm_params in m_params.items():
          if tm_key == 'posembed_input':  # Might need resolution change.
            posemb = params[b_key][m_key]['posembed_input']['pos_embedding']
            restored_posemb = m_params['posembed_input']['pos_embedding']

            if restored_posemb.shape != posemb.shape:
              # Rescale the grid of pos, embeddings: param shape is (1, N, d).
              logging.info('Resized variant: %s to %s', restored_posemb.shape,
                           posemb.shape)
              ntok = posemb.shape[1]
              if restored_model_cfg.model.classifier == 'token':
                # The first token is the CLS token.
                cls_tok = restored_posemb[:, :1]
                restored_posemb_grid = restored_posemb[0, 1:]
                ntok -= 1
              else:
                cls_tok = restored_posemb[:, :0]
                restored_posemb_grid = restored_posemb[0]

              restored_gs = int(np.sqrt(len(restored_posemb_grid)))
              gs = int(np.sqrt(ntok))
              if restored_gs != gs:  # We need resolution change.
                logging.info('Grid-size from %s to %s.', restored_gs, gs)
                restored_posemb_grid = restored_posemb_grid.reshape(
                    restored_gs, restored_gs, -1)
                zoom = (gs / restored_gs, gs / restored_gs, 1)
                restored_posemb_grid = scipy.ndimage.zoom(
                    restored_posemb_grid, zoom, order=1)
                restored_posemb_grid = restored_posemb_grid.reshape(
                    1, gs * gs, -1)
                # Attache the CLS token again.
                restored_posemb = jnp.array(
                    np.concatenate([cls_tok, restored_posemb_grid], axis=1))

            params[b_key][m_key][tm_key]['pos_embedding'] = restored_posemb
          else:  # Other parameters of the Transformer encoder.
            params[b_key][m_key][tm_key] = tm_params

      else:
        # Use the rest as they are in the pretrianed model.
        params[b_key][m_key] = m_params

  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))
