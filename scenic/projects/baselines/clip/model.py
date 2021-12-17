"""Provides builders and loaders of CLIP checkpoints."""

import os
from typing import Any, Mapping, Optional

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
from scenic.projects.baselines.clip import layers
from scenic.projects.baselines.clip import download

from tensorflow.io import gfile

# JAX team is working type checking for pytrees:
# https://github.com/google/jax/issues/3340
PyTree = Any

# pylint: disable=line-too-long
# Checkpoint paths from https://github.com/openai/CLIP/blob/main/clip/clip.py#L30
CHECKPOINTS_TORCH = {
    'resnet_50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt',
    'resnet_101': 'https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt',
    'resnet_50x4': 'https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt',
    'vit_b32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
    'vit_b16': 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
}

CHECKPOINTS = {
    'resnet_50': None,
    'resnet_101': None,
    'resnet_50x4': None,
    'vit_b32': None,
    'vit_b16': None,
}
# pylint: enable=line-too-long


MAX_TEXT_LENGTH = 77
IMAGE_RESOLUTION = 224
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711])

CONFIGS = {
    'vit_b32': dict(embed_dim=512,
                    vocab_size=49408,
                    vision_num_layers=12,
                    vision_features=768,
                    vision_patch_size=32,
                    text_features=512,
                    text_num_heads=8,
                    text_num_layers=12),
    'vit_b16': dict(embed_dim=512,
                    vocab_size=49408,
                    vision_num_layers=12,
                    vision_features=768,
                    vision_patch_size=16,
                    text_features=512,
                    text_num_heads=8,
                    text_num_layers=12),
    'resnet_50': dict(embed_dim=1024,
                      vocab_size=49408,
                      vision_num_layers=(3, 4, 6, 3),
                      vision_features=64,
                      text_features=512,
                      text_num_heads=8,
                      text_num_layers=12),
    'resnet_50x4': dict(embed_dim=640,
                        vocab_size=49408,
                        vision_num_layers=(4, 6, 10, 6),
                        vision_features=80,
                        text_features=640,
                        text_num_heads=10,
                        text_num_layers=12),
    'resnet_101': dict(embed_dim=512,
                       vocab_size=49408,
                       vision_num_layers=(3, 4, 23, 3),
                       vision_features=64,
                       text_features=512,
                       text_num_heads=8,
                       text_num_layers=12)
}


def load_model_vars(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    download_dir: str = download.DEFAULT_DOWNLOAD_DIR,
) -> PyTree:
  """Load model variables from a checkpoint, downloading if necessary."""
  checkpoint_path = checkpoint_path or CHECKPOINTS.get(model_name)
  if checkpoint_path is None:
   checkpoint_path = os.path.join(download_dir, model_name + '.npy')

   if not gfile.exists(checkpoint_path):
     # Download PyTorch checkpoint
     url = CHECKPOINTS_TORCH.get(model_name)
     logging.info('Downloading checkpoint from %s to %s', url, download_dir)
     checkpoint_path_torch = download.download(
         url, download_dir, expected_sha256=url.split('/')[-2])

     # Load and convert checkpoint to numpy
     logging.info('Converting checkpoint %s to numpy', checkpoint_path_torch)
     try:
       import torch
     except ImportError as e:
       logging.error('Could not import torch for CLIP checkpoint conversion')
     params = torch.jit.load(
         checkpoint_path_torch, map_location='cpu').state_dict()
     params = jax.tree_map(lambda p: p.cpu().numpy(), params)

     # Save converted checkpoint
     with gfile.GFile(checkpoint_path, 'wb') as f:
       np.save(f, params)
     del params
     gfile.remove(checkpoint_path_torch)

  with gfile.GFile(checkpoint_path, 'rb') as f:
    np_params = np.load(f, allow_pickle=True).tolist()
  return _convert_vars(np_params)


def vit_b32():
  return layers.CLIP(**CONFIGS['vit_b32'])


def vit_b16():
  return layers.CLIP(**CONFIGS['vit_b16'])


def resnet_50():
  return layers.CLIP(**CONFIGS['resnet_50'])


def resnet_50x4():
  return layers.CLIP(**CONFIGS['resnet_50x4'])


def resnet_101():
  return layers.CLIP(**CONFIGS['resnet_101'])


MODELS = {
    'resnet_50': resnet_50,
    'resnet_101': resnet_101,
    'resnet_50x4': resnet_50x4,
    'vit_b32': vit_b32,
    'vit_b16': vit_b16,
}


def _convert_attn_layers(params: Mapping[str, np.ndarray],
                         dim_head: int = 64) -> PyTree:
  """Convert attention parameters."""
  new_params = {}
  processed_attn_layers = []
  for k, v in params.items():
    if 'attn.' in k:
      base = k[:k.rindex('attn.')+5]
      if base in processed_attn_layers:
        continue
      processed_attn_layers.append(base)
      dim = params[base + 'out_proj.bias'].shape[-1]
      heads = dim // dim_head
      new_params[base + 'out.weight'] = params[
          base + 'out_proj.weight'].T.reshape(heads, dim_head, dim)
      new_params[base + 'out.bias'] = params[base + 'out_proj.bias']
      qkv_bias = params[base + 'in_proj_bias'].reshape(3, heads, dim_head)
      qkv_kernel = np.transpose(params[base + 'in_proj_weight'].reshape(
          3, heads, dim_head, dim), (0, 3, 1, 2))
      for i, kk in enumerate(('query', 'key', 'value')):
        new_params[base + f'{kk}.bias'] = qkv_bias[i]
        new_params[base + f'{kk}.weight'] = qkv_kernel[i]
    else:
      new_params[k] = v
  return new_params


def _convert_vars(torch_vars: Mapping[str, np.ndarray],
                  dim_head: int = 64) -> PyTree:
  """Convert torch parameters to flax parameters."""
  # Expand QKV dense input projection to separate Q, K, V projections
  # and fix shape/transposing of attention layers.
  torch_vars = _convert_attn_layers(torch_vars, dim_head)
  flax_vars = {}
  torch_vars.pop('context_length', None)
  torch_vars.pop('input_resolution', None)
  torch_vars.pop('vocab_size', None)
  for torch_key, v in torch_vars.items():
    if 'num_batches_tracked' in torch_key:
      continue

    if 'conv' in torch_key or 'downsample.0.weight' in torch_key:
      v = v.transpose(2, 3, 1, 0)
    elif 'weight' in torch_key and v.ndim == 2 and 'embedding' not in torch_key:
      # Fully connected layers are transposed, embeddings are not
      v = v.T

    jax_key = torch_key.replace('visual.proj', 'visual.proj.kernel')
    jax_key = jax_key.replace('text_projection', 'text_projection.kernel')
    if 'bn' in jax_key or 'ln' in jax_key or 'downsample.1' in jax_key:
      jax_key = jax_key.replace('.weight', '.scale')
    else:
      jax_key = jax_key.replace('.weight', '.kernel')
    if (jax_key.startswith('transformer') or
        jax_key.startswith('text_projection') or
        jax_key.startswith('ln_final') or
        jax_key.startswith('positional_embedding')):
      jax_key = 'text.' + jax_key

    jax_key = jax_key.replace(
        'token_embedding.kernel', 'text.token_embedding.embedding')

    jax_key = jax_key.replace('attnpool.k_proj', 'attnpool.attn.key')
    jax_key = jax_key.replace('attnpool.q_proj', 'attnpool.attn.query')
    jax_key = jax_key.replace('attnpool.v_proj', 'attnpool.attn.value')
    jax_key = jax_key.replace('attnpool.c_proj', 'attnpool.attn.out')
    if 'attnpool.attn.out' in jax_key:
      if jax_key.endswith('kernel'):
        v = v.reshape(-1, dim_head, v.shape[-1])
    elif 'attnpool.attn' in jax_key:
      if jax_key.endswith('bias'):
        v = v.reshape(-1, dim_head)
      else:
        v = v.reshape(v.shape[0], -1, dim_head)

    if jax_key.endswith('running_mean'):
      jax_key = 'batch_stats.' + jax_key.replace('.running_mean', '.mean')
    elif jax_key.endswith('running_var'):
      jax_key = 'batch_stats.' + jax_key.replace('.running_var', '.var')
    else:
      jax_key = 'params.' + jax_key

    jax_key = jax_key.replace('.', '/')
    jax_key = jax_key.replace('resblocks/', 'resblocks.')
    jax_key = jax_key.replace('resblocks/', 'resblocks.')

    flax_vars[tuple(jax_key.split('/'))] = jnp.asarray(v)

  # Transform the flattened param dict to the original nested structure.
  new_vars = flax.core.freeze(flax.traverse_util.unflatten_dict(flax_vars))
  return new_vars


def normalize_image(img: jnp.ndarray) -> jnp.ndarray:
  return (img - IMAGE_MEAN) / IMAGE_STD


def unnormalize_image(x: jnp.ndarray) -> jnp.ndarray:
  return x * IMAGE_STD + IMAGE_MEAN


# Class names and templates copied from:
# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
PROMPTS = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
