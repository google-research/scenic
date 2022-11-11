"""Utilities for DeformableDETR trainer/evaluator."""

from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from scenic.dataset_lib.coco_dataset import coco_eval
from scenic.projects.baselines.detr.train_utils import DetrGlobalEvaluator


class DetectionEvaluator(coco_eval.DetectionEvaluator):
  """Changes DETR Evaluator bay assuming labels vary in label_id space."""

  def __init__(self,
               annotations_loc: Optional[str] = None,
               threshold: float = 0.,
               disable_output: bool = True):
    """Initializes a DetectionEvaluator object."""
    super().__init__(
        annotations_loc=annotations_loc,
        threshold=threshold,
        disable_output=disable_output)

    # Just override the label map.
    max_id = max([c['id'] for c in self.coco.dataset['categories']])
    # Just a mapping from 0-index to 1-index, to remove no-object label.
    self.label_to_coco_id = {i: i + 1 for i in range(max_id)}


class DeformableDetrGlobalEvaluator(DetrGlobalEvaluator):
  """An interface between DeformableDETR implementation and COCO."""

  def __init__(self, dataset_name: str, **kwargs):
    """Instantiate evaluator and override to use DeformableDETR Eval."""
    del dataset_name  # Unused.

    self.coco_evaluator = DetectionEvaluator(**kwargs)
    self._included_image_ids = set()
    self._num_examples_added = 0


def prepare_coco_eval_dicts(
    batch: Dict[str, Any],
    predictions: Dict[str, jnp.ndarray],
    logits_to_probs_fn: Callable[[jnp.ndarray], jnp.ndarray],
    gather: bool = False
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
  """Prepare predictions and validations for COCO eval.

  Args:
    batch: Eval batch targets.
    predictions: Predictions from DeformableDETR.
    logits_to_probs_fn: Convert logits to probabilities.
    gather: Whether to perform gather if we are in a pmapped eval.

  Returns:
    Targets and predictions formatted for COCO eval.
  """
  pred_probs = logits_to_probs_fn(predictions['pred_logits'])
  # Collect necessary predictions and target information from all hosts.
  predictions_out = {
      'pred_probs': pred_probs,
      'pred_logits': predictions['pred_logits'],
      'pred_boxes': predictions['pred_boxes']
  }
  labels = {
      'image/id': batch['label']['image/id'],
      'size': batch['label']['size'],
      'orig_size': batch['label']['orig_size'],
  }
  to_copy = [
      'labels', 'boxes', 'not_exhaustive_category_ids', 'neg_category_ids'
  ]
  for name in to_copy:
    if name in batch['label']:
      labels[name] = batch['label'][name]

  targets = {'label': labels, 'batch_mask': batch['batch_mask']}

  if gather:
    predictions_out = jax.lax.all_gather(predictions_out, 'batch')
    targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions_out
