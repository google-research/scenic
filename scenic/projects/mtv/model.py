"""Implements the MTV model."""
import functools
from typing import Any, Callable, Iterable, List, MutableSequence, Optional, Sequence, Tuple, Union

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.mfp import mvit
from scenic.projects.mtv import model_utils
from scenic.projects.vivit import model as vivit_model
from scenic.train_lib_deprecated import train_utils

_DEFAULT_MTV_CONFIG = ml_collections.ConfigDict({
    'dataset_configs': {
        'num_frames': 8,
    },
    'model':
        dict(
            view_configs=[
                ml_collections.ConfigDict({
                    'hidden_size': 16,
                    'patches': {
                        'size': (4, 4, 2)
                    },
                    'num_heads': 2,
                    'mlp_dim': 32,
                    'num_layers': 1,
                })
            ],
            cross_view_fusion=None,
            temporal_encoding_config=ml_collections.ConfigDict({
                'method': '3d_conv',
                'kernel_init_method': 'central_frame_initializer',
            }),
            global_encoder_config=ml_collections.ConfigDict({
                'num_layers': 2,
                'mlp_dim': 8,
                'num_heads': 2,
                'hidden_size': 8,
            }),
            dropout_rate=0.,
            attention_dropout_rate=0.,
            classifier='token',
            data_dtype_str='float32')
})


def get_model_cls(model_name):
  """"Selects MTV model type."""
  if model_name == 'mtv_multiclass_classification':
    return MTVClassificationModel
  elif model_name == 'mtv_multihead_classification':
    return MTVMultiheadClassificationModel
  else:
    raise ValueError('Unrecognized model: {}'.format(model_name))


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    This class is branched from
    https://github.com/google/flax/blob/main/flax/linen/attention.py.
    The difference is that we added an option to zero-initialize the output
    projection layer. We use this trick to insert cross attention layers inside
    a MTV model without disturbing the pretrained weights. Note that if we
    simply initialize all Query, Key, Value, and Out matrices to zeros, their
    weights will not be updated during training.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      param_dtype: the dtype passed to parameter initializers (default:
        float32).
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      qkv_kernel_init: initializer for the kernel of the Dense layers.
      out_kernel_init: initializer for the kernel of the output projection
        layers.
      qkv_bias_init: initializer for the bias of the Dense layers.
      out_bias_init: initializer for the bias of the output projection.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
  """
  num_heads: int
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: Any = None
  qkv_kernel_init: Callable[[Any, Iterable[int], Any],
                            jnp.ndarray] = nn.initializers.xavier_uniform()
  out_kernel_init: Callable[[Any, Iterable[int], Any],
                            jnp.ndarray] = nn.initializers.xavier_uniform()
  qkv_bias_init: Callable[[Any, Iterable[int], Any],
                          jnp.ndarray] = nn.initializers.zeros
  out_bias_init: Callable[[Any, Iterable[int], Any],
                          jnp.ndarray] = nn.initializers.zeros
  use_bias: bool = True

  @nn.compact
  def __call__(self,
               inputs_q: jnp.ndarray,
               inputs_kv: jnp.ndarray,
               mask: Optional[jnp.ndarray] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape `[batch_sizes..., length, features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = nn.merge_param('deterministic', self.deterministic,
                                     deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.qkv_kernel_init,
        bias_init=self.qkv_bias_init,
        use_bias=self.use_bias,
        precision=self.precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    x = nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)
    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        batch_dims=(),
        kernel_init=self.out_kernel_init,
        bias_init=self.out_bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name='out')(
            x)
    return out


class CrossViewAttentionEncoderBlock(nn.Module):
  """Crossview Transformer encoder layer.

  The encoder architecture for each view is as follows:
  Layer norm
  cross attention (out projection weights are initialized with zeros)
  residual connection
  Layer norm
  self attention (initialized with pretrained ViT weights)
  residual connection
  Layer norm
  MLP (initialized with pretrained ViT weights)
  residual connection

  We apply cross attention in a sequential fashion and limit it to only take
  place in neighboring views. For example, view[i-1] is used as the query and
  view[i] is used as key and value. This design is based on the assumption
  that the tubelet sizes grow from 0th view to the nth view. We initialize cross
  attention's weights with zeros and self attention and MLP weights are
  initialized with pretrained ViTs.

  Attributes:
    view_configs: Model configs for each view (e.g., num_heads, mlp_dim, etc).
    cross_view_fusion: Cross view fusion config.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """
  num_layers: Sequence[int]
  mlp_widths: Sequence[int]
  view_configs: Sequence[ml_collections.ConfigDict]
  cross_view_fusion: ml_collections.ConfigDict
  dtype: Any = jnp.float32
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  def get_stochastic_depth_mask(self, lyr: int, num_layers: int, x: jnp.ndarray,
                                deterministic: bool) -> jnp.ndarray:
    """Generate the stochastic depth mask in order to apply layer-drop.

    Args:
      lyr: The current layer.
      num_layers: Number of layers in total.
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Stochastic depth mask.
    """
    stochastic_depth = (lyr / max(num_layers - 1, 1)) * self.stochastic_depth
    if not deterministic and stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), stochastic_depth, shape)
    else:
      return 0.0

  def _apply_self_attentions(self, tokens: List[jnp.ndarray], cur_layer: int,
                             deterministic: bool) -> List[jnp.ndarray]:
    """Applies self attentions for each view."""
    for view_idx, x in enumerate(tokens):
      if cur_layer >= self.num_layers[view_idx]:
        continue
      y = nn.LayerNorm(dtype=self.dtype, name=f'msa_ln_view{view_idx}')(x)
      config = self.view_configs[view_idx]
      y = nn.MultiHeadDotProductAttention(
          num_heads=config['num_heads'],
          dtype=self.dtype,
          broadcast_dropout=False,
          deterministic=deterministic,
          dropout_rate=self.attention_dropout_rate,
          name=f'msa_view{view_idx}')(y, y)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic)
      tokens[view_idx] += y * (1.0 - self.get_stochastic_depth_mask(
          cur_layer, self.num_layers[view_idx], y,
          deterministic))
    return tokens

  def _apply_cross_attention(
      self,
      tokens: List[jnp.ndarray],
      cur_layer: int,
      num_heads_per_view: Sequence[int],
      deterministic: bool,
      fuse_in_descending_order: bool,
  ) -> List[jnp.ndarray]:
    """Applies cross view attention."""
    xs = [
        nn.LayerNorm(dtype=self.dtype, name=f'cross_attention_ln_view{idx}')(x)
        for idx, x in enumerate(tokens)
    ]
    view_indices = (
        range(len(xs) -
              1, 0, -1) if fuse_in_descending_order else range(len(xs) - 1))
    for view_index in view_indices:
      query_view_index = (
          view_index - 1 if fuse_in_descending_order else view_index + 1)
      key_value_view_index = view_index
      query = xs[query_view_index]
      key_value = xs[key_value_view_index]
      num_heads = (
          num_heads_per_view[query_view_index]
          if self.cross_view_fusion.use_query_config else
          num_heads_per_view[key_value_view_index])
      qkv_features = (
          query.shape[-1]
          if self.cross_view_fusion.use_query_config else key_value.shape[-1])

      y = MultiHeadDotProductAttention(
          num_heads=num_heads,
          dtype=self.dtype,
          qkv_features=qkv_features,
          qkv_kernel_init=nn.initializers.xavier_uniform(),
          out_kernel_init=nn.initializers.zeros,
          broadcast_dropout=False,
          deterministic=deterministic,
          dropout_rate=self.attention_dropout_rate,
          name=f'cross_attention_view{query_view_index}_{key_value_view_index}'
      )(query, key_value)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic)
      tokens[query_view_index] += (
          y * (1.0 - self.get_stochastic_depth_mask(
              cur_layer, self.num_layers[query_view_index], y, deterministic)))
    return tokens

  def _apply_mlp(self, tokens: List[jnp.ndarray], cur_layer: int,
                 deterministic: bool) ->List[jnp.ndarray]:
    """Applies MLP block."""
    for view_idx, x in enumerate(tokens):
      if cur_layer >= self.num_layers[view_idx]:
        continue
      y = nn.LayerNorm(dtype=self.dtype, name=f'mlp_ln_view{view_idx}')(x)
      y = attention_layers.MlpBlock(
          mlp_dim=self.mlp_widths[view_idx],
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          activation_fn=nn.gelu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name=f'mlp_view{view_idx}')(
              y, deterministic=deterministic)
      tokens[view_idx] += (
          y * (1.0 - self.get_stochastic_depth_mask(
              cur_layer, self.num_layers[view_idx], y, deterministic)))
    return tokens

  @nn.compact
  def __call__(self, tokens: List[jnp.ndarray], cur_layer: int,
               num_heads_per_view: Sequence[int],
               deterministic: bool,
               skip_mlp_and_selfattention: bool = False) -> List[jnp.ndarray]:
    """Applies CrossViewAttentionEncoderBlock module.

    Args:
      tokens: Input tokens from each view.
      cur_layer: Which layer we apply cross attention.
      num_heads_per_view: Number of attention heads in each view.
      deterministic: Deterministic or not (to apply dropout).
      skip_mlp_and_selfattention: If true, skip the MLP and SelfAttn.

    Returns:
      Output tokens for each view.
    """
    tokens = self._apply_cross_attention(
        tokens, cur_layer, num_heads_per_view, deterministic,
        self.cross_view_fusion.get('fuse_in_descending_order', True))
    if not skip_mlp_and_selfattention:
      tokens = self._apply_self_attentions(tokens, cur_layer, deterministic)
      tokens = self._apply_mlp(tokens, cur_layer, deterministic)
    return tokens


class MultiviewEncoder(nn.Module):
  """Multiview Transformer Encoder.

  Attributes:
    view_configs: Model configs for each view (e.g., num_heads, mlp_dim, etc).
    cross_view_fusion: Cross view fusion config.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows the
      timm library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Any of activations.
  """
  view_configs: Sequence[ml_collections.ConfigDict]
  cross_view_fusion: ml_collections.ConfigDict
  input_token_temporal_dims: Sequence[int]
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  def _split_tokens_and_bottleneck(
      self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Removes bottleneck tokens from input."""
    return (tokens[:, :-self.cross_view_fusion.bottleneck_tokens],
            tokens[:, -self.cross_view_fusion.bottleneck_tokens:])

  def _add_posembed(self, tokens: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
    """Adds positional embeddings."""
    temporal_dims_after_alignment = [
        t // min(self.input_token_temporal_dims)
        for t in self.input_token_temporal_dims
    ]
    xs = []
    for idx, t in enumerate(tokens):
      bs, spacetime, channels = t.shape
      reshaped_t = t.reshape(
          (bs, temporal_dims_after_alignment[idx], -1, channels))
      add_posembed_fn = vit.AddPositionEmbs(name=f'posembed_input_view{idx}')
      x = jax.vmap(add_posembed_fn, in_axes=1, out_axes=1)(reshaped_t)
      xs.append(x.reshape(bs, spacetime, channels))
    return xs

  def _build_with_bottleneck(
      self,
      xs: List[jnp.ndarray],
      bottleneck: jnp.ndarray,
      fusion_layers: Sequence[int],
      max_num_layers: int,
      train: bool,
      dtype: Any,
  ) -> List[jnp.ndarray]:
    """Builds the encoder with bottlenecks."""
    view_indices = list(range(len(self.view_configs)))
    if self.cross_view_fusion.get('fuse_in_descending_order', True):
      view_indices.reverse()
    for lyr in range(max_num_layers):
      for view_idx in view_indices:
        view_config = self.view_configs[view_idx]
        if lyr >= view_config['num_layers']:
          continue
        if lyr in fusion_layers:
          if xs[view_idx].shape[-1] != bottleneck.shape[-1]:
            bottleneck = nn.Dense(
                xs[view_idx].shape[-1],
                kernel_init=nn.initializers.xavier_uniform(),
                name=f'bottleneck_linear_{lyr}_view{view_idx}')(
                    bottleneck)
          xs[view_idx] = jnp.concatenate([xs[view_idx], bottleneck], axis=1)

        xs[view_idx] = vit.Encoder1DBlock(
            mlp_dim=view_config['mlp_dim'],
            num_heads=view_config['num_heads'],
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=(lyr / max(view_config['num_layers'] - 1, 1)) *
            self.stochastic_depth,
            name=f'encoderblock_{lyr}_view{view_idx}',
            dtype=self.dtype)(
                xs[view_idx], deterministic=not train)
        if lyr in fusion_layers:
          xs[view_idx], bottleneck = self._split_tokens_and_bottleneck(
              xs[view_idx])
    return xs

  def _build_with_cross_view_attention(
      self,
      xs: List[jnp.ndarray],
      fusion_layers: Sequence[int],
      max_num_layers: int,
      train: bool,
      dtype: Any,
  ) -> List[jnp.ndarray]:
    """Builds the encoder with bottlenecks."""
    num_heads_per_view = [v['num_heads'] for v in self.view_configs]
    for lyr in range(max_num_layers):
      if lyr in fusion_layers:
        xs = CrossViewAttentionEncoderBlock(
            num_layers=[v['num_layers'] for v in self.view_configs],
            mlp_widths=[v['mlp_dim'] for v in self.view_configs],
            view_configs=self.view_configs,
            cross_view_fusion=self.cross_view_fusion,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            name=f'cross_view_encoderblock_{lyr}')(
                xs, lyr, num_heads_per_view, deterministic=not train)
      else:
        for view_idx, view_config in enumerate(self.view_configs):
          if lyr >= view_config['num_layers']:
            continue
          xs[view_idx] = vit.Encoder1DBlock(
              mlp_dim=view_config['mlp_dim'],
              num_heads=view_config['num_heads'],
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=(lyr / max(view_config['num_layers'] - 1, 1)) *
              self.stochastic_depth,
              name=f'encoderblock_{lyr}_view{view_idx}',
              dtype=self.dtype)(
                  xs[view_idx], deterministic=not train)
    return xs

  def _build_with_cross_view_attention_mvit(
      self,
      xs: MutableSequence[jnp.ndarray],
      fusion_layers: Sequence[int],
      spacetime_shapes: Any,
      train: bool,
      dtype: Any,
  ) -> MutableSequence[jnp.ndarray]:
    """Builds the encoder with bottlenecks."""

    num_heads_per_view = [
        v.get('num_initial_heads', 1) for v in self.view_configs
    ]
    stride_kv = [v.initial_kv_pooling_strides[:] for v in self.view_configs]
    num_layers = [v.depth for v in self.view_configs]

    for layer_idx in range(max(num_layers)):
      if layer_idx in fusion_layers:
        xs = CrossViewAttentionEncoderBlock(
            num_layers=num_layers,
            mlp_widths=[v.width*4 for v in self.view_configs],
            view_configs=self.view_configs,
            cross_view_fusion=self.cross_view_fusion,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            name=f'cross_view_encoderblock_{layer_idx}')(
                xs,
                layer_idx,
                num_heads_per_view,
                train,
                skip_mlp_and_selfattention=True)

      for view_idx, view_config in enumerate(self.view_configs):
        if layer_idx >= view_config['depth']:
          continue

        x = xs[view_idx]
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        logging.info('view %d, layer %d shape %s (spacetime: %s',
                     view_idx, layer_idx, x.shape, spacetime_shapes[view_idx])
        out_dim = x.shape[-1]
        pool_q = view_config.pooling_kernel
        if layer_idx in view_config.pooling_layers:
          num_heads_per_view[view_idx] *= 2
          stride_kv[view_idx] = [
              (x // 2) if x > 1 else 1 for x in stride_kv[view_idx]
          ]
          stride_q = view_config.pooling_strides_q
        else:
          stride_q = [1 for _ in view_config.pooling_strides_q]
          if not view_config.pool_q_every_layer:
            pool_q = [1 for _ in view_config.pooling_strides_q]

        if layer_idx + 1 in view_config.pooling_layers:
          out_dim = x.shape[-1] * 2

        stoch_depth = (layer_idx / max(view_config.depth - 1,
                                       1)) * view_config.stochastic_depth

        attn_kwargs = mvit.convert_attention_kwargs(
            view_config.get('attention_kwargs', {}))
        x, spacetime_shapes[view_idx] = mvit.Block(
            num_heads=num_heads_per_view[view_idx],
            out_dim=out_dim,
            pooling_kernel_q=pool_q,
            pooling_kernel_kv=view_config.pooling_kernel,
            pooling_stride_q=stride_q,
            pooling_stride_kv=stride_kv[view_idx],
            attention_kwargs=attn_kwargs,
            stochastic_depth=stoch_depth,
            name=f'view{view_idx}_block_{layer_idx}',
            use_bias=view_config.get('use_bias', False),
        )(x, spacetime_shapes[view_idx], deterministic=not train)
        xs[view_idx] = x
    return xs

  @nn.compact
  def __call__(self,
               tokens: MutableSequence[jnp.ndarray],
               bottleneck: Union[jnp.ndarray, None],
               spacetime_shapes: Any,
               train: bool = False) -> MutableSequence[jnp.ndarray]:
    """Applies Transformer model on the tokens.

    This function will be called within a vmap along the time axis. Before
    calling this function, we need to make sure all elements in the list have
    the same temporal dimension.

    Args:
      tokens: A sequence of input tubelet tokens. Each one is a 3D float tensor
        of shape (batch, sequence_len, channels). We assume that tokens[0]
        contains tokens from the largest view while tokens[-1] are from the
        smallest view. We define a view as a representation of the input video
        composed of tubelets. A larger view corresponds to larger tubelets.
      bottleneck: A 3D float tensor of shape (batch, num_tokens, channels)
        representing a set of tokens used for fusing information among views.
      spacetime_shapes: spatiotemoral shapes of the tokens.
      train: Whether or not it is in training.

    Returns:
      A list of activations after encoding for each view. They have the same
      shapes as their input counterparts.
    """

    for t in tokens:
      assert t.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    fusion_layers = ([] if self.cross_view_fusion is None else
                     self.cross_view_fusion.fusion_layers)

    is_mvit = 'patch_size' in self.view_configs[0]
    if is_mvit:  # MViT
      return self._build_with_cross_view_attention_mvit(
          tokens, fusion_layers, spacetime_shapes, train, dtype)

    xs = self._add_posembed(tokens)
    max_num_layers = max([config['num_layers'] for config in self.view_configs])
    if (self.cross_view_fusion is None or
        self.cross_view_fusion.type == 'cross_view_attention'):
      return self._build_with_cross_view_attention(xs, fusion_layers,
                                                   max_num_layers, train, dtype)
    if self.cross_view_fusion.type == 'bottleneck':
      assert not is_mvit, 'Bottlenecks not implemented in MTV-MViT'
      return self._build_with_bottleneck(xs, bottleneck, fusion_layers,
                                         max_num_layers, train, dtype)
    raise ValueError(
        f'Invalid cross view fusion type: {self.cross_view_fusion.type}.')


class MTV(nn.Module):
  """MTV model."""
  view_configs: Sequence[ml_collections.ConfigDict]
  cross_view_fusion: ml_collections.ConfigDict
  temporal_encoding_config: ml_collections.ConfigDict
  global_encoder_config: ml_collections.ConfigDict
  input_token_temporal_dims: Sequence[int]
  num_classes: int
  classifier: str
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  keep_spatiotemporal_features: bool = False
  final_endpoint: str = 'logits'
  dtype: Any = jnp.float32

  def _add_cls_token(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Prepends CLS token.

    Args:
      x: A 3D float tensor of shape (batch, sequence_len, channels) representing
        the tokens.
      name: Parameter name of the added CLS token.

    Returns:
      A 3D float tensor with prepended CLS token. Its new shape is (batch,
      sequence_len+1, channels).
    """
    if self.classifier == 'token':
      bs, _, c = x.shape
      cls = self.param(name, nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [bs, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    return x

  def _add_cls_tokens_all_frames(self, x: jnp.ndarray,
                                 name: str) -> jnp.ndarray:
    """Prepends CLS token for all frames.

    Args:
      x: A 4D float tensor of shape (batch, time, sequence_len, channels)
        representing the tokens.
      name: Parameter name of the added CLS token.

    Returns:
      A 4D float tensor with prepended CLS token. Its new shape is (batch, time,
      sequence_len+1, channels).
    """
    if self.classifier == 'token':
      bs, time, _, c = x.shape
      cls = self.param(name, nn.initializers.zeros, (1, time, 1, c), x.dtype)
      cls = jnp.tile(cls, [bs, 1, 1, 1])
      x = jnp.concatenate([cls, x], axis=2)
    return x

  def _add_cls_tokens_for_all_views(
      self, tokens: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
    """Prepends CLS tokens for all views.

    Args:
      tokens: Tokens from all views. Each one has a shape of (batch, time,
        sequence_len, channels)

    Returns:
      A list of tokens with CLS tokens added. Each one has a new shape of
      (batch, time, sequence_len+1, channels).
    """
    outputs = []
    for idx, x in enumerate(tokens):
      outputs.append(self._add_cls_tokens_all_frames(x, name=f'cls_view{idx}'))
    return outputs

  def _extract_encoder_output(self,
                              x: jnp.ndarray,
                              axis: int = 1) -> jnp.ndarray:
    """Extracts encoder output."""
    if self.classifier in ['token', '0']:
      x = x.take(indices=0, axis=axis)
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=list(range(axis, x.ndim - 1)))
    return x

  def _tokenize(
      self, x: jnp.ndarray
  ) -> Tuple[MutableSequence[jnp.ndarray], MutableSequence[jnp.ndarray]]:
    """Creates tokens for each view.

    Args:
      x: A 5D float tensor of shape (batch, time, height, width, channels)
        representing the input video.

    Returns:
      Tokens for each view and each one has a shape of (batch, time,
      sequence_len, channels), and a list with the original shapes of
      each token.
    """
    tokens = []
    shapes = []
    for idx, config in enumerate(self.view_configs):
      if 'patch_size' in config:  # MViT.
        view_tokens = mvit.create_patches(x, config.width, config.patch_size,
                                          config.patch_stride,
                                          f'view{idx}_embedding',
                                          padding='SAME')
        view_tokens = mvit.add_position_embedding(view_tokens, 'learned',
                                                  f'view{idx}_posemb_kernel')
      else:
        view_tokens, _ = vivit_model.temporal_encode(
            x,
            self.temporal_encoding_config,
            ml_collections.ConfigDict(config['patches']),
            config['hidden_size'],
            return_1d=False,
            name=f'embedding_view{idx}')
      shapes.append(view_tokens.shape)
      bs, t, h, w, c = view_tokens.shape
      view_tokens = view_tokens.reshape(bs, t, h * w, c)
      tokens.append(view_tokens)
    return tokens, shapes

  def _align_temporal_dimension_across_views(
      self, tokens: MutableSequence[jnp.ndarray],
      spacetime_shapes: MutableSequence[MutableSequence[int]]
  ) -> Tuple[MutableSequence[jnp.ndarray],
             MutableSequence[MutableSequence[int]]]:
    """Reshapes tokens from each view so they have the same temporal dim."""
    min_temporal_dim = min(self.input_token_temporal_dims)
    outputs = []
    for i, t in enumerate(tokens):
      n, time, wh, c = t.shape
      seq_len = (wh * time) // min_temporal_dim
      outputs.append(t.reshape(n, min_temporal_dim, seq_len, c))
      spacetime_shapes[i][0] = time // min_temporal_dim
    return outputs, spacetime_shapes

  def _merge_views_along_time_axis(self, tokens: Sequence[jnp.ndarray],
                                   hidden_size: int) -> jnp.ndarray:
    """Merges tokens from each view along the time axis."""
    projected_tokens = []
    for view_idx, x in enumerate(tokens):
      bs, time, n, c = x.shape
      x = x.reshape(bs, self.input_token_temporal_dims[view_idx],
                    (time * n) // self.input_token_temporal_dims[view_idx], c)
      if not self.keep_spatiotemporal_features:
        x = self._extract_encoder_output(x, axis=2)
      projected_tokens.append(
          nn.Dense(
              hidden_size,
              kernel_init=nn.initializers.xavier_uniform(),
              name=f'global_encoder_linear_view{view_idx}')(x))
    return jnp.concatenate(projected_tokens, axis=1)

  def _merge_views_along_channel_axis(
      self, tokens: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Merges tokens from each view along the channel axis."""
    max_temporal_dim = max(self.input_token_temporal_dims)
    xs = []
    for idx, x in enumerate(tokens):
      bs, time, n, c = x.shape
      x = x.reshape(bs, self.input_token_temporal_dims[idx],
                    (time * n) // self.input_token_temporal_dims[idx], c)
      if self.keep_spatiotemporal_features:
        xs.append(jnp.tile(x, (1, max_temporal_dim // x.shape[1], 1, 1)))
      else:
        x = self._extract_encoder_output(x, axis=2)
        xs.append(jnp.tile(x, (1, max_temporal_dim // x.shape[1], 1)))
    return jnp.concatenate(xs, axis=-1)

  def _global_encode(self, tokens: MutableSequence[jnp.ndarray],
                     is_train: bool) -> jnp.ndarray:
    """Applies the global encoder.

    We support two strategies to merge encoded tokens from each view:

    In the first strategy, we extract the CLS tokens from each view (we apply
    pooling when other classifiers are used), apply tiling to match the temporal
    dimension, and concatenate them in the channel dimension.

    In the second strategy, after we extract the CLS tokens we linear project
    them into the same dimension and concatenate them along the temporal
    dimension.

    The global encoder is implemented as a ViT encoder.

    Args:
      tokens: A list of tokens from each view. Each one has a shape of (batch,
        time, sequence_len, channels).
      is_train: Whether or not it is in training.

    Returns:
      A 2D float tensor representing the embedding from the global encoder.
    """
    encoder_config = self.global_encoder_config.to_dict()
    encoder_config.update(
        dict(
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            dtype=self.dtype,
            name='global_encoder'))
    merge_axis = encoder_config.pop('merge_axis', 'channel')
    hidden_size = encoder_config.pop('hidden_size')
    if merge_axis == 'time':
      x = self._merge_views_along_time_axis(tokens, hidden_size)
    elif merge_axis == 'channel':
      x = self._merge_views_along_channel_axis(tokens)
    else:
      raise ValueError(f'Invalid merge_axis: {merge_axis}.')
    x = self._add_cls_token(x, name='cls_global')
    encoder = vit.Encoder(**encoder_config)
    if self.keep_spatiotemporal_features:
      x = jax.vmap(
          functools.partial(encoder, train=is_train), in_axes=2, out_axes=2)(
              x)
    else:
      x = encoder(x, train=is_train)
    return (x if self.keep_spatiotemporal_features else
            self._extract_encoder_output(x))

  def _encode_per_time(
      self,
      tokens: MutableSequence[jnp.ndarray],
      bottleneck: Union[jnp.ndarray, None],
      spacetime_shapes: Sequence[int],
      is_train: bool,
  ) -> MutableSequence[jnp.ndarray]:
    """Encodes input tokens on a per-time basis."""

    tokens = MultiviewEncoder(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion,
        input_token_temporal_dims=self.input_token_temporal_dims,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='MultiviewEncoder')(
            tokens, bottleneck=bottleneck, train=is_train,
            spacetime_shapes=spacetime_shapes)
    return tokens

  def _check_config(self, x: jnp.ndarray):
    """Checks configuration errors."""
    if 'patch_size' in self.view_configs[0]:  # MViT.
      return
    if self.keep_spatiotemporal_features and self.classifier == 'token':
      raise ValueError('Classifier cannot be `token` when '
                       '`keep_spatiotemporal_features` is True.')
    heights = [config['patches']['size'][0] for config in self.view_configs]
    widths = [config['patches']['size'][1] for config in self.view_configs]
    if self.keep_spatiotemporal_features and (len(set(heights)) > 1 or
                                              len(set(widths)) > 1):
      raise ValueError('Patches from different views must have the same height '
                       'and width when `keep_spatiotemporal_features` is True.')

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool = True,
               debug: bool = False):
    """Executes MTV model.

    Args:
      x: A 5D float tensor of shape (batch, time, height, width, channels)
        representing the input video.
      train: Whether or not it is in training.
      debug: Whether or not it is in debug mogde. Not used here.

    Returns:
      The logits produced by the MTV model.
    """
    del debug
    self._check_config(x)
    tokens, shapes = self._tokenize(x)
    logging.info('token shapes after tokenizing: %s', [t.shape for t in tokens])
    spacetime_shapes = [list(s[1:-1]) for s in shapes]
    if 'patch_size' not in self.view_configs[0]:  # ViViT.
      tokens = self._add_cls_tokens_for_all_views(tokens)
    tokens, spacetime_shapes = self._align_temporal_dimension_across_views(
        tokens, spacetime_shapes)
    logging.info('token shapes after aligning: %s (spacetime: %s)',
                 [t.shape for t in tokens], spacetime_shapes)

    if (self.cross_view_fusion is not None and
        self.cross_view_fusion.type == 'bottleneck'):
      if self.cross_view_fusion.get('fuse_in_descending_order', True):
        channels = tokens[-1].shape[-1]
      else:
        channels = tokens[0].shape[-1]
      bottleneck = self.param(
          'bottleneck', nn.initializers.normal(stddev=0.02),
          (1, tokens[0].shape[1], self.cross_view_fusion.bottleneck_tokens,
           channels), self.dtype)
      bottleneck = jnp.tile(bottleneck, [x.shape[0], 1, 1, 1])
      tokens = jax.vmap(
          functools.partial(self._encode_per_time, is_train=train,
                            spacetime_shapes=spacetime_shapes),
          in_axes=(1, 1),
          out_axes=1)(tokens, bottleneck)
    else:
      tokens = jax.vmap(
          functools.partial(
              self._encode_per_time, bottleneck=None,
              is_train=train, spacetime_shapes=spacetime_shapes),
          in_axes=1,
          out_axes=1)(tokens)
    logging.info('token shapes after per_time_encoding: %s',
                 [t.shape for t in tokens])
    tokens = self._global_encode(tokens, train)
    logging.info('token shapes after global encoding: %s',
                 [t.shape for t in tokens])
    if self.keep_spatiotemporal_features:
      bs, _, h, w, _ = x.shape
      tokens = tokens.reshape(
          (bs, tokens.shape[1], h // self.view_configs[0].patches.size[0],
           w // self.view_configs[0].patches.size[1], -1))
    pre_logits = nn_layers.IdentityLayer(name='pre_logits')(tokens)
    if self.final_endpoint == 'pre_logits':
      return pre_logits
    if self.keep_spatiotemporal_features:
      pre_logits = self._extract_encoder_output(pre_logits, axis=1)
    logits = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            pre_logits)
    if self.final_endpoint == 'logits':
      return logits
    raise ValueError(f'Final endpoint `{self.final_endpoint}` not recognized.')


class MTVClassificationModel(vivit_model.ViViTClassificationModel):
  """MTV model for multiclass classification task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return MTV(
        input_token_temporal_dims=model_utils.get_input_token_temporal_dims(
            self.config.dataset_configs.num_frames,
            self.config.model.view_configs),
        num_classes=self.dataset_meta_data['num_classes'],
        dtype=model_dtype, **self.config.model)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _DEFAULT_MTV_CONFIG

  def init_from_train_state(
      self,
      train_state: train_utils.TrainState,
      restored_train_state: train_utils.TrainState,
      restored_model_cfg: ml_collections.ConfigDict,
      restore_output_proj: bool = False) -> train_utils.TrainState:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. The input
    embeddings and positional embeddings are resized if the current model uses
    a different size of tubelets than the pretrained model.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.
      restore_output_proj: Whether or not to restore output projection weights.

    Returns:
      Updated train_state.
    """
    return model_utils.initialize_from_mtv_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_projection=restore_output_proj)

  def init_from_vit_train_states(
      self,
      train_state: train_utils.TrainState,
      restored_train_states: Sequence[train_utils.TrainState],
      restored_model_cfgs: Sequence[ml_collections.ConfigDict],
      restored_model_formats: Sequence[str],
  ) -> train_utils.TrainState:
    """Updates the train_state with data from restored_train_states.

    This function is used to initialize a MTV model from a list of ViT
    checkpoints. We assume that the number of restored_train_states is equal to
    the number of views.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_states: A sequence of TrainStates that is loaded with
        parameters/state of a pretrained ViT model.
      restored_model_cfgs: A sequence of model configuration of the pretrained
        ViT models. Usually used for some asserts.
      restored_model_formats: The checkpoint format of each model. The format
        can be 'scenic' or 'big_vision'.

    Returns:
      Updated train_state.
    """
    return model_utils.initialize_from_vit_train_states(self.config,
                                                        train_state,
                                                        restored_train_states,
                                                        restored_model_cfgs,
                                                        restored_model_formats)

  def init_from_mvit_train_states(
      self,
      train_state: train_utils.TrainState,
      restored_train_states: Sequence[train_utils.TrainState],
      restored_model_cfgs: Sequence[ml_collections.ConfigDict],
  ) -> train_utils.TrainState:
    """Updates the train_state with data from restored_train_states.

    This function is used to initialize a MTV model from a list of MViT
    checkpoints. We assume that the number of restored_train_states is equal to
    the number of views.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_states: A sequence of TrainStates that is loaded with
        parameters/state of a pretrained ViT model.
      restored_model_cfgs: A sequence of model configuration of the pretrained
        ViT models. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return model_utils.initialize_from_mvit_train_states(self.config,
                                                         train_state,
                                                         restored_train_states,
                                                         restored_model_cfgs)


class MTVMultiheadClassificationModel(
    vivit_model.ViViTMultiHeadClassificationModel, MTVClassificationModel):
  """MTV model for multi-classification tasks.

  When methods are overriden by both parents, the implementation follows the
  first parent, which is ViViTMultiHeadClassificationModel in this case. For
  build_flax_model() and default_flax_model_config(), we explicitly call the
  methods from MTVClassificationModel.
  """

  def build_flax_model(self) -> nn.Module:
    return MTVClassificationModel.build_flax_model(self)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return MTVClassificationModel.default_flax_model_config(self)
