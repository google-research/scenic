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

"""Utilities for DETR trainer."""

import copy
import json
import os
from typing import Any, Dict, Optional, Set

from absl import logging
from flax import core as flax_core
from flax import optim as optimizers
from flax import traverse_util
import jax
from jax.example_libraries import optimizers as experimental_optimizers
import jax.numpy as jnp
import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont
from scenic.common_lib import image_utils
from scenic.dataset_lib.coco_dataset import coco_eval
from scenic.model_lib.base_models import box_utils
from scenic.train_lib_deprecated import optimizers as scenic_optimizers
from scenic.train_lib_deprecated import train_utils
import scipy.special
import tensorflow as tf


class DetrGlobalEvaluator():
  """An interface between the Scenic DETR implementation and COCO evaluators."""

  def __init__(self, dataset_name: str, **kwargs):
    del dataset_name  # Unused.

    self.coco_evaluator = coco_eval.DetectionEvaluator(
        threshold=0.0, **kwargs)
    self._included_image_ids = set()
    self._num_examples_added = 0

  def add_example(
      self, prediction: Dict[str, np.ndarray], target: Dict[str, np.ndarray]):
    """Add a single example to the evaluator.

    Args:
      prediction: Model prediction dictionary with keys 'pred_img_ids',
        'pred_probs' in shape of `[num_objects, num_classes]` and 'pred_boxes'
        in shape of `[num_objects, 4]`. Box coordinates should be in raw DETR
        format, i.e. [cx, cy, w, h] in range [0, 1].
      target: Target dictionary with keys 'orig_size', 'size', and 'image/id'.
        Must also contain 'padding_mask' if the input image was padded.
    """
    if 'pred_boxes' not in prediction:
      # Add dummy to make eval work:
      prediction = copy.deepcopy(prediction)
      prediction['pred_boxes'] = np.zeros(
          (prediction['pred_logits'].shape[0], 4)) + 0.5

    # Convert from DETR [cx, cy, w, h] to COCO [x, y, w, h] bounding box format:
    boxes = box_utils.box_cxcywh_to_xyxy(prediction['pred_boxes'])
    boxes = np.array(boxes)
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    # Scale from relative to absolute size:
    # Note that the padding is implemented such that such that the model's
    # predictions are [0,1] normalized to the non-padded image, so scaling by
    # `orig_size` will convert correctly to the original image coordinates. No
    # image flipping happens during evaluation.
    h, w = np.asarray(target['orig_size'])
    scale_factor = np.array([w, h, w, h])
    boxes = boxes * scale_factor[np.newaxis, :]
    boxes_np = np.asarray(boxes)

    # Get scores, excluding the background class:
    if 'pred_probs' in prediction:
      scores = prediction['pred_probs'][:, 1:]
    else:
      scores = scipy.special.softmax(prediction['pred_logits'], axis=-1)[:, 1:]

    # Add example to evaluator:
    self.coco_evaluator.add_annotation(
        bboxes=boxes_np,
        scores=np.asarray(scores),
        img_id=int(target['image/id']))

    self._num_examples_added += 1

  def compute_metrics(
      self,
      included_image_ids: Optional[Set[int]] = None,
      clear_annotations: Optional[bool] = True) -> Dict[str, Any]:
    """Computes the metrics for all added predictions."""
    if included_image_ids is not None:
      self.coco_evaluator.coco.reload_ground_truth(included_image_ids)
    return self.coco_evaluator.compute_coco_metrics(
        clear_annotations=clear_annotations)

  def clear(self):
    self.coco_evaluator.clear_annotations()
    self._num_examples_added = 0

  def __len__(self):
    return self._num_examples_added

  def write_pred_annotations_to_file(self,
                                     path: str,
                                     fname_app: Optional[str] = None,
                                     clear_annotations: Optional[bool] = True):
    """Writes predictions to file in JSON format.

    Args:
      path: Path to write the prediction annotation JSON file.
      fname_app: Optional string to append to the file name.
      clear_annotations: Clear annotations after they are stored in a file.
    """
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    json_file_name = f"predictions{fname_app if fname_app else ''}.json"
    json_file_path = os.path.join(path, json_file_name)

    def _convert_to_serializable(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.float32):
        return float(obj)
      else:
        raise TypeError(f'Unserializable object {obj} of type {type(obj)}')

    with tf.io.gfile.GFile(json_file_path, 'w') as f:
      f.write(
          json.dumps(
              self.coco_evaluator.annotations,
              default=_convert_to_serializable))
    logging.info('Predicted annotations are stored in %s.', json_file_path)
    if clear_annotations:
      self.coco_evaluator.clear_annotations()


def _unpad_and_resize(masks, padding_mask, orig_size):
  """Removes padding and resizes masks."""
  # Resize masks to the padding_mask size in case they have a lower resolution:
  masks = image_utils.resize_pil(
      masks,
      out_h=padding_mask.shape[0],
      out_w=padding_mask.shape[1],
      num_batch_dims=1,
      method='linear')
  # Remove padding:
  row_masks = np.any(padding_mask, axis=-1)
  col_masks = np.any(padding_mask, axis=-2)
  masks = masks[:, row_masks, :]
  masks = masks[:, :, col_masks]
  # Resize to original size:
  return image_utils.resize_pil(
      masks,
      out_h=orig_size[0],
      out_w=orig_size[1],
      num_batch_dims=1,
      method='linear')


def normalize_metrics_summary(metrics_summary, split,
                              object_detection_loss_keys):
  """Normalizes the metrics in the given metrics summary.

  Note that currently we only support metrics of the form 1/N sum f(x_i).

  Args:
    metrics_summary: dict; Each value is a sum of a calculated metric over all
      examples.
    split: str; Split for which we normalize the metrics. Used for logging.
    object_detection_loss_keys: list; A loss key used for computing the object
      detection loss.

  Returns:
    Normalized metrics summary.

  Raises:
    TrainingDivergedError: Due to observing a NaN in the metrics.
  """
  for key, val in metrics_summary.items():
    metrics_summary[key] = val[0] / val[1]
    if np.isnan(metrics_summary[key]):
      raise train_utils.TrainingDivergedError(
          'NaN detected in {}'.format(f'{split}_{key}'))

  # compute and add object_detection_loss using globally normalize terms
  object_detection_loss = 0
  for loss_term_key in object_detection_loss_keys:
    object_detection_loss += metrics_summary[loss_term_key]
  metrics_summary['object_detection_loss'] = object_detection_loss

  return metrics_summary


def process_and_fetch_to_host(pred_or_tgt, batch_mask):
  """Used to collect predictions and targets of the whole valid/test set.

  Args:
    pred_or_tgt: pytree; A pytree of jnp-arrays where leaves are of shape
      `[num_devices, bs, X,...,Y]`.
    batch_mask: A nd-array of shape `[num_devices, bs]`, where zero values
      indicate padded examples.

  Returns:
    A list of length num_devices * bs of items, where each item is a tree with
    the same structure as `pred_or_tgt` and each leaf contains a single example.
  """
  # Fetch to host in a single call.
  pred_or_tgt, batch_mask = jax.device_get((pred_or_tgt, batch_mask))
  batch_mask = np.array(batch_mask).astype(bool)

  def _split_mini_batches(x):
    # Filter out padded examples.
    x = x[batch_mask]
    # Split minibatch of examples into a list of examples.
    x_list = np.split(x, x.shape[0], axis=0)
    # Squeeze out the dummy dimension.
    return jax.tree_util.tree_map(lambda x: np.squeeze(x, axis=0), x_list)

  leaves, treedef = jax.tree_util.tree_flatten(pred_or_tgt)

  batch_shape = batch_mask.shape
  assert all([leaf.shape[:2] == batch_shape for leaf in leaves]), (
      'Inconsistent batch shapes.')

  # Split batched leaves into lists of examples:
  leaves = list(map(_split_mini_batches, leaves))

  # Go from leaf-lists to list of trees:
  out = []
  if leaves:
    num_examples = np.sum(batch_mask, dtype=np.int32)
    for example_ind in range(num_examples):
      out.append(treedef.unflatten([leaf[example_ind] for leaf in leaves]))
  return out


def draw_boxes_side_by_side(pred: Dict[str, Any], batch: Dict[str, Any],
                            label_map: Dict[int, str]) -> np.ndarray:
  """Side-by-side visualization of detection predictions and ground truth."""

  viz = []

  # unnormalizes images to be [0, 1]
  mean_rgb = np.reshape(np.array([0.48, 0.456, 0.406]), [1, 1, 1, 3])
  std_rgb = np.reshape(np.array([0.229, 0.224, 0.225]), [1, 1, 1, 3])
  imgs = ((batch['inputs'] * std_rgb + mean_rgb) * 255.0).astype(np.uint8)

  font = PIL.ImageFont.load_default()

  # iterates over images in the batch and makes visualizations
  for indx in range(imgs.shape[0]):
    h, w = batch['label']['size'][indx]

    # first for ground truth
    gtim = PIL.Image.fromarray(imgs[indx])
    gtdraw = PIL.ImageDraw.Draw(gtim)
    for bb, cls, is_crowd in zip(batch['label']['boxes'][indx],
                                 batch['label']['labels'][indx],
                                 batch['label']['is_crowd'][indx]):
      if cls == 0:
        continue  # dummy object.

      bcx, bcy, bw, bh = bb * [w, h, w, h]
      bb = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
      if is_crowd:
        edgecolor = (255, 0, 0)
      else:
        edgecolor = (255, 255, 0)

      gtdraw.rectangle(bb, fill=None, outline=edgecolor, width=3)
      gtdraw.text([bb[0], max(bb[1] - 10, 0)],
                  label_map[cls],
                  font=font,
                  fill=(0, 0, 255))

    # second for model predictions
    predim = PIL.Image.fromarray(imgs[indx])
    preddraw = PIL.ImageDraw.Draw(predim)
    pred_lbls = np.argmax(pred['pred_logits'], axis=-1)
    for bb, cls in zip(pred['pred_boxes'][indx], pred_lbls[indx]):
      h, w = batch['label']['size'][indx]
      bcx, bcy, bw, bh = bb * [w, h, w, h]
      bb = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
      edgecolor = (0, 255, 0) if cls else (0, 150, 0)
      preddraw.rectangle(
          bb, fill=None, outline=edgecolor, width=3 if cls else 1)
    # Separate loop for text to prevent occlusion of text by boxes:
    for bb, cls in zip(pred['pred_boxes'][indx], pred_lbls[indx]):
      if not cls:
        continue
      h, w = batch['label']['size'][indx]
      bcx, bcy, bw, bh = bb * [w, h, w, h]
      bb = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2]
      preddraw.text([bb[0], max(bb[1] - 10, 0)],
                    label_map[cls],
                    font=font,
                    fill=(0, 0, 255))

    gtim_np = np.asarray(gtim)
    predim_np = np.asarray(predim)
    composite = np.concatenate([predim_np, gtim_np], axis=1)

    viz.append(composite)
  return np.stack(viz, axis=0)


def get_detr_optimizer(config):
  """Makes a Flax MultiOptimizer for DETR."""
  other_optim = scenic_optimizers.get_optimizer(config)

  if config.get('backbone_training'):
    backbone_optim = scenic_optimizers.get_optimizer(config.backbone_training)
  else:
    backbone_optim = other_optim

  def is_bn(path):
    # For DETR we need to skip the BN affine transforms as well.
    names = ['/bn1/', '/bn2/', '/bn3/', '/init_bn/', '/proj_bn/']
    for s in names:
      if s in path:
        return True
    return False
  backbone_traversal = optimizers.ModelParamTraversal(
      lambda path, param: ('backbone' in path) and (not is_bn(path)))
  other_traversal = optimizers.ModelParamTraversal(
      lambda path, param: 'backbone' not in path)

  return MultiOptimizerWithLogging((backbone_traversal, backbone_optim),
                                   (other_traversal, other_optim))


class MultiOptimizerWithLogging(optimizers.MultiOptimizer):
  """Adds logging to MultiOptimizer to show which params are trained."""

  def init_state(self, params):
    self.log(params)
    return super().init_state(params)

  def log(self, inputs):
    for i, traversal in enumerate(self.traversals):
      params = _get_params_dict(inputs)
      flat_dict = traverse_util.flatten_dict(params)
      for key, value in _sorted_items(flat_dict):
        path = '/' + '/'.join(key)
        if traversal._filter_fn(path, value):  # pylint: disable=protected-access
          logging.info(
              'ParamTraversalLogger (opt %d): %s, %s', i, value.shape, path)


def _sorted_items(x):
  """Returns items of a dict ordered by keys."""
  return sorted(x.items(), key=lambda x: x[0])


def _get_params_dict(inputs):
  if isinstance(inputs, (dict, flax_core.FrozenDict)):
    return flax_core.unfreeze(inputs)
  else:
    raise ValueError(
        'Can only traverse a flax Model instance or a nested dict, not '
        f'{type(inputs)}')


def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  # jax.example_libraries.optimizers.clip_grads implements this differently.
  # First, it uses clip_coef of max_norm / norm without the 1e-6.
  # Second, the jnp.where condition and argument order are reversed. This should
  # normally be a no-change but we do not know what changes in XLA are triggered
  # as a result of this and how that effects precision etc.
  norm = experimental_optimizers.l2_norm(grad_tree)
  clip_coef = max_norm / (norm + 1e-6)
  normalize = lambda g: jnp.where(clip_coef < 1., g * clip_coef, g)
  return jax.tree_util.tree_map(normalize, grad_tree)
