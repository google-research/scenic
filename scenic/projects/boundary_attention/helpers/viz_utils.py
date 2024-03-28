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

"""Utilities for visualization."""

import einops
import jax.numpy as jnp
from matplotlib import colors
import matplotlib.pyplot as plt


def visualize_outputs(input_image, outputs, num_levels=8):
  """Script to visualize the outputs of the network."""

  my_pubu = plt.get_cmap('PuBu')(jnp.arange(plt.get_cmap('PuBu').N))
  my_pubu[:, 0:3] *= 0.7
  my_pubu = colors.ListedColormap(my_pubu)

  # Crop the outputs to account for boundary effects
  global_distances = outputs[-1]['global_distances'].squeeze()[8:-8, 8:-8]
  jn_levels = jnp.linspace(0, jnp.max(global_distances), num_levels)

  # Define mesh of pixel coordinates in width x height
  xymesh = jnp.meshgrid(jnp.arange(0.5, global_distances.shape[1] + 0.5, 1.0),
                        jnp.arange(0.5, global_distances.shape[0] + 0.5, 1.0))

  # Threshold output boundaries
  output_boundaries = (outputs[-1]['global_boundaries'].squeeze() * (
      outputs[-1]['global_boundaries'].squeeze() > .3))[8:-8, 8:-8]
  output_features = outputs[-1]['global_features'].squeeze(
      ).transpose(1, 2, 0)[8:-8, 8:-8, :]

  plt.figure(figsize=(20, 10))
  plt.subplot(141)
  plt.imshow(input_image.squeeze().transpose(1, 2, 0)[8:-8, 8:-8])
  plt.axis('off')
  plt.subplot(142)
  plt.imshow(global_distances, cmap='PuBu')
  plt.contour(xymesh[0],
              xymesh[1],
              global_distances,
              levels=jn_levels,
              cmap=my_pubu)
  plt.axis('off')
  plt.subplot(143)
  plt.imshow(output_boundaries, cmap='gray')
  plt.axis('off')
  plt.subplot(144)
  plt.imshow(output_features)
  plt.axis('off')
  plt.tight_layout()
  plt.show()


def make_weight_map(params2maps, yc, xc, patchres, height, width, hpatches,
                    wpatches, wedges, crop=False):
  """Make a weight map for a given pixel.

  Args:
    params2maps: a map of params to maps
    yc: y coordinate
    xc: x coordinate
    patchres: patch resolution
    height: height
    width: width
    hpatches: number of patches
    wpatches: number of patches
    wedges: wedges
    crop: whether to crop the image

  Returns:
    A weight map for the pixel
  """

  # This defines where the query pixel is located depending on which patch it
  # belongs to
  x, y = jnp.meshgrid(jnp.arange(patchres-1, -1, -1),
                      jnp.arange(patchres-1, -1, -1))

  # This is the patch for each pixel
  p_yc_range = jnp.minimum(jnp.maximum(yc - y, 0), hpatches).astype(int)
  p_xc_range = jnp.minimum(jnp.maximum(xc - x, 0), wpatches).astype(int)

  # Make a binary indicator array indicating whether wedge m contains pixel
  # [xc, yc]
  bin_ind = jnp.zeros_like(wedges)
  bin_array = bin_ind.at[:, :, :, :, p_yc_range,
                         p_xc_range].set(jnp.expand_dims(
                             wedges[:, :, y, x, p_yc_range, p_xc_range],
                             (2, 3)))

  bin_array = jnp.sum(bin_array * wedges, axis=1, keepdims=True)
  binary_wedge_map = params2maps.fold(bin_array,
                                      (1, 1, height, width)).squeeze()
  binary_wedge_map = binary_wedge_map/jnp.max(binary_wedge_map)

  if crop:
    binary_wedge_map_padded = jnp.pad(binary_wedge_map,
                                      ((patchres, patchres),
                                       (patchres, patchres)))
    binary_wedge_map = binary_wedge_map_padded[yc:yc+patchres*2,
                                               xc:xc+patchres*2]

  return binary_wedge_map


def patchstack(patches, border=2, padvalue=0.0):
  """Stack field of patches into one large image.

  Args:
    patches: a tensor of patches
    border: space (in pixels) between neighboring patches (integer)
    padvalue: value to fill space with

  Returns:
    A tensor of stacked patches
  """

  assert border % 2 == 0, f'border must be even (but got {border})'

  # Pad 3rd and 4th to last dimensions with border//2 pixels valued `padvalue`.
  padamt = ((0, 0),
            (0, 0),
            (border//2, border//2),
            (border//2, border//2),
            (border//2, border//2),
            (border//2, border//2))

  padded = jnp.pad(patches, padamt, mode='constant', constant_values=padvalue)

  output = einops.rearrange(padded, 'b c p1 p2 hp wp -> b (hp p1) (wp p2) c')

  return output


def image_to_uint8(float_image: jnp.ndarray, mean=None, stddev=None):
  """Converts a float image to uint8. Also un-whitens the image, if needed."""
  if stddev is not None:
    float_image = float_image * stddev
  if mean is not None:
    float_image = float_image + mean
  if mean is None and stddev is None:
    float_image = float_image * 255.
  float_image = jnp.round(jnp.clip(float_image, 0., 255.))
  return float_image.astype(jnp.uint8)


def get_viz_dict_from_batch(batch, model_outputs, model, name,
                            num_image_summaries=16):
  """Get a dictionary of images to be written to disk.

  Args:
    batch: a batch of data
    model_outputs: a dictionary of model outputs
    model: a model
    name: a string
    num_image_summaries: number of image summaries to generate

  Returns:
    Arrays containing different visualizations.
  """
  def get_images(config, model_outputs, im_num, ii, sbatch):

    if config.model_name == 'deformable_boundary_attention_v0' or (
        config.model_name == 'deformable_boundary_attention') or (
            config.model_name == 'boundary_attention'):

      global_features = model_outputs[ii]['global_features'][sbatch,
                                                             im_num, ...]
      global_distances = model_outputs[ii]['global_distances'][sbatch,
                                                               im_num, ...]
      global_distances = global_distances/jnp.max(global_distances,
                                                  axis=(2, 3), keepdims=True)
      global_boundaries = model_outputs[ii]['global_boundaries'][sbatch,
                                                                 im_num, ...]
      output_pred_scales = model_outputs[ii]['patchsize_distribution'][sbatch,
                                                                       im_num,
                                                                       ...]
    else:
      raise NotImplementedError('Need to define visualization for model.')

    return global_distances, global_boundaries, global_features, (
        output_pred_scales)

  # accumulate image dictionary
  num_image_summaries = min(num_image_summaries, batch['image'].shape[1])

  num_iters_plot = min(3, len(model_outputs))
  iters = jnp.linspace(-num_iters_plot,
                       -num_iters_plot+num_iters_plot-1,
                       num_iters_plot).astype(int)
  sbatch = jnp.arange(1)

  write_images = {}
  for nn in range(num_image_summaries):
    for ii in iters:

      token_name = '%s_sample%d_iter%d' % (name, nn, ii)

      # (shard_batch, iter, scale, batch, feature, H, W)
      input_opts = model.config.model.input_opts

      input_image = (batch['image'][sbatch, nn, ...])
      input_boundaries = (batch['boundaries'][sbatch, nn, ...])
      input_boundaries = input_boundaries/jnp.max(input_boundaries,
                                                  axis=(2, 3), keepdims=True)
      input_distances = (batch['distances'][sbatch, nn, ...])
      input_distances = input_distances/jnp.max(input_distances,
                                                axis=(2, 3), keepdims=True)

      global_distances, global_boundaries, global_features, (
          output_pred_scales) = get_images(model.config, model_outputs, nn,
                                           ii, sbatch)

      # White out the border pixels
      # Make border pixles
      border_template = jnp.ones_like(global_distances)
      hps = input_opts.patchsize // 2
      border_template = border_template.at[:, :, hps, hps:-hps].set(0)
      border_template = border_template.at[:, :, -hps, hps:-hps].set(0)
      border_template = border_template.at[:, :, hps:-hps, hps].set(0)
      border_template = border_template.at[:, :, hps:-hps, -hps].set(0)

      input_image = input_image*border_template + (1-border_template)
      input_boundaries = input_boundaries*border_template + (
          1-border_template)
      input_distances = input_distances*border_template + (
          1-border_template)
      output_distances = global_distances*border_template + (
          1-border_template)
      output_boundaries = global_boundaries*border_template + (
          1-border_template)
      output_global_features = global_features*border_template + (
          1- border_template)

      # frames
      write_images['%s/input_image' % token_name] = (
          image_to_uint8(input_image.transpose(0, 2, 3, 1)))
      write_images['%s/input_boundaries' % token_name] = (
          image_to_uint8(input_boundaries.transpose(0, 2, 3, 1)))
      write_images['%s/input_distances' % token_name] = (
          image_to_uint8(input_distances.transpose(0, 2, 3, 1)))
      write_images['%s/output_distances' % token_name] = (
          image_to_uint8(output_distances.transpose(0, 2, 3, 1)))
      write_images['%s/output_boundaries' % token_name] = (
          image_to_uint8(output_boundaries.transpose(0, 2, 3, 1)))
      write_images['%s/output_global_features' % token_name] = (
          image_to_uint8(output_global_features.transpose(0, 2, 3, 1)))

      if output_pred_scales is not None:
        all_data = []
        for jj in range(output_pred_scales.shape[0]):
          fig = plt.figure()
          plt.imshow(output_pred_scales[jj])
          plt.axis('off')
          plt.colorbar(fraction=0.046, pad=0.04, shrink=.95)
          fig.tight_layout(pad=0)
          fig.canvas.draw()
          plt.close(fig)

          data = jnp.frombuffer(fig.canvas.tostring_rgb(), dtype=jnp.uint8)
          data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

          all_data.append(data)

        write_images['%s/output_pred_scale' % token_name] = (
            jnp.stack(all_data, axis=0))

  return write_images

