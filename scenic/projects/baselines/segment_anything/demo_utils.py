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

"""Util functions for running the SAM demo."""

import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from PIL import Image

from scenic.projects.baselines.segment_anything.modeling import sam
from tensorflow.io import gfile


def get_encoder_config(model_size):
  dim, depth, num_heads, dp, window_block_indexes = sam.SIZE_CONFIGS[model_size]
  image_encoder_args = ml_collections.ConfigDict()
  image_encoder_args['embed_dim'] = dim
  image_encoder_args['depth'] = depth
  image_encoder_args['num_heads'] = num_heads
  image_encoder_args['drop_path_rate'] = dp
  image_encoder_args['window_block_indexes'] = window_block_indexes
  return image_encoder_args


def load_image(image_path):
  image = np.array(
      Image.open(gfile.GFile(image_path, 'rb')), dtype=np.uint8).copy()
  return image


def resize_and_pad_image(image, target_size=1024):
  h, w = image.shape[:2]
  scale = 1.0 * target_size / max(h, w)
  new_h, new_w = int(scale * h + 0.5), int(scale * w + 0.5)
  image = np.array(
      Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR))
  ret = np.zeros((1, target_size, target_size, 3))
  padding_mask = np.zeros((1, target_size, target_size), np.float32)
  ret[0, :image.shape[0], :image.shape[1]] = image
  padding_mask[0, :image.shape[0], :image.shape[1]] = 1
  return ret, padding_mask, (h, w)


def get_point_coords_and_labels(point_prompts, input_size, ori_size):
  ori_h, ori_w = ori_size
  point_coords = np.asarray(
      point_prompts, dtype=np.float32).reshape(1, 1, -1, 2)
  point_coords[..., 0] = point_coords[..., 0] / max(ori_h, ori_w) * input_size
  point_coords[..., 1] = point_coords[..., 1] / max(ori_h, ori_w) * input_size
  point_labels = np.ones(point_coords.shape[:-1], dtype=np.int32)
  return point_coords, point_labels


def show_mask(mask, ax, random_color=False):
  if random_color:
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
  else:
    color = np.array([30/255, 144/255, 255/255, 0.6])
  h, w = mask.shape[-2:]
  mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
  pos_points = coords[labels == 1]
  neg_points = coords[labels == 0]
  ax.scatter(
      pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
      s=marker_size, edgecolor='white', linewidth=1.25)
  ax.scatter(
      neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
      s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
  x0, y0 = box[0], box[1]
  w, h = box[2] - box[0], box[3] - box[1]
  ax.add_patch(
      plt.Rectangle(
          (x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plot(
    image, point_coords, point_labels, mask, score, figsize=10, save_path=''):
  plt.figure(figsize=(figsize, figsize))
  plt.imshow(image.astype(np.uint8))
  show_mask(mask, plt.gca())
  show_points(point_coords[0], point_labels[0], plt.gca())
  plt.title(f'Score: {score:.3f}', fontsize=18)
  plt.axis('off')
  if save_path:
    plt.savefig(gfile.GFile(save_path, 'w'))
  else:
    plt.show()


def plot_all_masks(image, ret, figsize=20, save_path=''):
  """Plots all masks."""
  plt.figure(figsize=(figsize, figsize))
  plt.imshow(image.astype(np.uint8))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  img = np.ones((ret['masks'].shape[1], ret['masks'].shape[2], 4))
  img[:, :, 3] = 0
  for i in range(ret['iou_predictions'].shape[0]):
    if ret['iou_predictions'][i] > 0:
      m = np.asarray(ret['masks'][i])
      color_mask = np.concatenate([np.random.random(3), [0.35]])
      img[m] = color_mask
  ax.imshow(img)
  plt.axis('off')
  if save_path:
    plt.savefig(gfile.GFile(save_path, 'w'))
  else:
    plt.show()
