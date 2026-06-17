"""Implementation of MLP-Mixer model."""

from typing import Any, Dict, Sequence

from absl import logging
import flax
import flax.linen as nn
from gvt.models import vqvae
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
import scipy


class MixerBlock(nn.Module):
  """Mixer block consisting of a token- and a channel-mixing phase.

  Attributes:
    channels_mlp_dim: Hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: Hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: The layer dropout rate (= stochastic depth).

  Returns:
    Output after mixer block.
  """
  channels_mlp_dim: int
  sequence_mlp_dim: int
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

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

  # Having this as a separate function makes it possible to capture the
  # intermediate representation via capture_intermediandarrates.
  def combine_branches(self, long_branch: jnp.ndarray,
                       short_branch: jnp.ndarray) -> jnp.ndarray:
    """Merges residual connections."""
    return long_branch + short_branch

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies the Mixer block to inputs."""
    if inputs.ndim != 3:
      raise ValueError('Input should be of shape `[batch, tokens, channels]`.')

    # Token mixing part, provides between-patches communication.
    x = nn.LayerNorm()(inputs)
    x = jnp.swapaxes(x, 1, 2)

    x = attention_layers.MlpBlock(
        mlp_dim=self.sequence_mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='token_mixing')(
            x, deterministic=deterministic)

    x = jnp.swapaxes(x, 1, 2)
    x *= 1.0 - self.get_stochastic_depth_mask(x, deterministic)
    x = self.combine_branches(x, inputs)

    # Channel-mixing part, which provides within-patch communication.
    y = nn.LayerNorm()(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.channels_mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='channel_mixing')(
            y, deterministic=deterministic)

    y *= 1.0 - self.get_stochastic_depth_mask(y, deterministic)
    return self.combine_branches(y, x)


class MixerBackbone(nn.Module):
  """Mixer model.

  Attributes:
    num_classes: Number of output classes.
    patch_size: Patch size of the stem.
    hidden_size: Size of the hidden state of the output of model's stem.
    num_layers: Number of layers.
    channels_mlp_dim: hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: overall stochastic depth rate.
  """

  num_classes: int
  patch_size: Sequence[int]
  hidden_size: int
  num_layers: int
  channels_mlp_dim: int
  sequence_mlp_dim: int
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  in_downsample: int = 1
  is_fusion: bool = False
  fusion_dim: int = 128

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               x_pixel: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False) -> jnp.ndarray:

    if self.is_fusion:
      n, gh, gw, c = x.shape
      fh, fw = self.patch_size

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
      x = nn.Conv(
          self.hidden_size,
          (self.in_downsample, self.in_downsample),
          strides=(self.in_downsample, self.in_downsample),
          padding='VALID',
          name='linear_projection')(
              x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    for i in range(self.num_layers):
      p = (i / max(self.num_layers - 1, 1)) * self.stochastic_depth
      x = MixerBlock(
          channels_mlp_dim=self.channels_mlp_dim,
          sequence_mlp_dim=self.sequence_mlp_dim,
          dropout_rate=self.dropout_rate,
          stochastic_depth=p,
          name=f'mixerblock_{i}')(
              x, deterministic=not train)
    x = nn.LayerNorm(name='pre_logits_norm')(x)
    # Use global average pooling for classifier:
    x = jnp.mean(x, axis=1)
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    return nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)


class VQMixer(nn.Module):
  """Mixer model.

  Attributes:
    num_classes: Number of output classes.
    patch_size: Patch size of the stem.
    hidden_size: Size of the hidden state of the output of model's stem.
    num_layers: Number of layers.
    channels_mlp_dim: hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: overall stochastic depth rate.
  """

  config: ml_collections.ConfigDict
  dataset_meta_data: Dict[str, Any]

  def setup(self):
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    self.vqgan = vqvae.VQVAE(self.config, train=False, dtype=model_dtype)
    self.vit = MixerBackbone(
        num_classes=self.dataset_meta_data['num_classes'],
        patch_size=self.config.model.patch_size,
        hidden_size=self.config.model.hidden_size,
        num_layers=self.config.model.num_layers,
        channels_mlp_dim=self.config.model.channels_mlp_dim,
        sequence_mlp_dim=self.config.model.sequence_mlp_dim,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        in_downsample=self.config.get('in_downsample', 1),
        is_fusion=self.config.get('is_fusion', False),
        fusion_dim=self.config.get('fusion_dim', 128))

    if self.config.get('retrain_embed_code', False):
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

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               rng_tmp: Any = None,
               *,
               train: bool,
               debug: bool = False) -> jnp.ndarray:

    x_vq_input = {'image': (x + 1.0) / 2.0}
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
    return self.vit(embedding, x_pixel=x, train=train)


class VQMixerMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Mixer model for multi-label classification task."""

  def build_flax_model(self) -> nn.Module:
    return VQMixer(config=self.config, dataset_meta_data=self.dataset_meta_data)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                patch_size=(4, 4),
                hidden_size=16,
                num_layers=1,
                channels_mlp_dim=32,
                sequence_mlp_dim=32,
                dropout_rate=0.,
                stochastic_depth=0,
            )
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
  # if 'g_optimizer' in restored_train_state:
  #   restored_params = flax.core.unfreeze(
  #       restored_train_state['g_optimizer']['target'])
  # else:
  restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)

  # Start moving parameters, one-by-one and apply changes if needed.
  print('restored_params', restored_params)
  for b_key, b_params in restored_params.items():
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

