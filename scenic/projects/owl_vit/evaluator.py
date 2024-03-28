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

r"""Run LVIS evaluation.

This script runs inference on a TFDS dataset (by default, the LVIS validation
set), writes the predictions to disk in the LVIS JSON format, and runs the LVIS
API evaluation on the files.

The ground-truth annotations must be supplied in the LVIS JSON format in the
local directory or at --annotations_path. The official annotations can be
obtained at https://www.lvisdataset.org/dataset.

The model is specified via --checkpoint_path and a --config matching the model.

See flag definitions in code for advanced settings.

Example command:
python evaluator.py \
  --alsologtostderr=true \
  --config=clip_b32 \
  --output_dir=/tmp/evaluator

"""
# GOOGLE INTERNAL pylint: disable=g-importing-member
import collections
import datetime
import functools
import json
import multiprocessing
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib
import zipfile

from absl import app
from absl import flags
from absl import logging
from clu import preprocess_spec
from flax import linen as nn
import jax
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
from lvis.eval import LVISEval
from lvis.lvis import LVIS
from lvis.results import LVISResults
from matplotlib import pyplot as plt
import ml_collections
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.preprocessing import image_ops
from scenic.projects.owl_vit.preprocessing import label_ops
from scenic.projects.owl_vit.preprocessing import modalities
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

LVIS_VAL_URL = 'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip'

COCO_METRIC_NAMES = [
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
]

_DEFAULT_ANNOTATIONS_PATH = '~/annotations/lvis_v1_val.json'

flags.DEFINE_string(
    'config',
    None,
    'Name of the config of the model to use for inference.',
    required=True)
flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Checkpoint path to use. Must match the model in the config.',
    required=True)
flags.DEFINE_string(
    'output_dir', None, 'Directory to write predictions to.', required=True)
flags.DEFINE_string(
    'tfds_name',
    'lvis',
    'TFDS name of the dataset to run inference on.')
flags.DEFINE_string('split', 'validation', 'Dataset split to run inference on.')
flags.DEFINE_string(
    'annotations_path',
    _DEFAULT_ANNOTATIONS_PATH,
    'Path to JSON file with ground-truth annotations in COCO/LVIS format. '
    'If it does not exist, the script will try to download it.')
flags.DEFINE_enum('data_format', 'lvis', ('lvis', 'coco'),
                  'Whether to use the LVIS or COCO API.')
flags.DEFINE_enum('platform', 'cpu', ('cpu', 'gpu', 'tpu'), 'JAX platform.')
flags.DEFINE_string(
    'tfds_data_dir', None,
    'TFDS data directory. If the dataset is not available in the directory, it '
    'will be downloaded.'
    )
flags.DEFINE_string(
    'tfds_download_dir', None,
    'TFDS download directory. Defaults to ~/tensorflow-datasets/downloads.')
flags.DEFINE_integer(
    'num_example_images_to_save', 10,
    'Number of example images with predictions to save.')
flags.DEFINE_integer(
    'label_shift', 1,
    'Value that will be added to the model output labels in the prediction '
    'JSON files. The model predictions are zero-indexed. COCO or LVIS use '
    'one-indexed labels, so label_shift should be 1 for these datasets. Set '
    'it to 0 for zero-indexed datasets.'
)

FLAGS = flags.FLAGS

_MIN_BOXES_TO_PLOT = 5
_PRED_BOX_PLOT_FACTOR = 3
_PLOTTING_SCORE_THRESHOLD = 0.01


Variables = nn.module.VariableDict
ModelInputs = Any
Predictions = Any


def _timestamp() -> str:
  return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')


def get_dataset(tfds_name: str,
                split: str,
                input_size: int,
                tfds_data_dir: Optional[str] = None,
                tfds_download_dir: Optional[str] = None,
                data_format: str = 'lvis') -> Tuple[tf.data.Dataset, List[str]]:
  """Returns a tf.data.Dataset and class names."""
  builder = tfds.builder(tfds_name, data_dir=tfds_data_dir)
  builder.download_and_prepare(download_dir=tfds_download_dir)
  class_names = builder.info.features['objects']['label'].names
  ds = builder.as_dataset(split=split)
  if data_format == 'lvis':
    decoder = image_ops.DecodeLvisExample()
  elif data_format == 'coco':
    decoder = image_ops.DecodeCocoExample()
  else:
    raise ValueError(f'Unknown data format: {data_format}.')
  pp_fn = preprocess_spec.PreprocessFn([
      decoder,
      image_ops.ResizeWithPad(input_size, pad_value=0.0),
      image_ops.Keep(
          [modalities.IMAGE, modalities.IMAGE_ID, modalities.ORIGINAL_SIZE])
  ], only_jax_types=True)
  ds = (
      ds.map(pp_fn, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(1)
      .batch(jax.device_count())
      .prefetch(tf.data.AUTOTUNE)
  )
  return ds, class_names


def tokenize_queries(tokenize: Callable[[str, int], List[int]],
                     queries: List[str],
                     prompt_template: str = '{}',
                     max_token_len: int = 16) -> List[List[int]]:
  """Tokenizes a sequence of query strings.

  Args:
    tokenize: Tokenization function.
    queries: List of strings to embed.
    prompt_template: String with '{}' placeholder to use as prompt template.
    max_token_len: If the query+prompt has more tokens than this, it will be
      truncated.

  Returns:
    A list of lists of tokens.
  """
  return [
      tokenize(
          label_ops._canonicalize_string_py(prompt_template.format(q)),  # pylint: disable=protected-access
          max_token_len) for q in queries
  ]


def get_embed_queries_fn(
    module: nn.Module,
    variables: Variables) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Get query embedding function.

  Args:
    module: OWL-ViT Flax module.
    variables: OWL-ViT variables.

  Returns:
    Jitted query embedding function.
  """

  @jax.jit
  def embed(queries):
    return module.apply(
        variables,
        text_queries=queries,
        train=False,
        method=module.text_embedder)

  return embed


def get_predict_fn(
    module: nn.Module,
    variables) -> Callable[[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
  """Get prediction function.

  Args:
    module: OWL-ViT Flax module.
    variables: OWL-ViT variables.

  Returns:
    Jitted predict function.
  """

  def apply(method, **kwargs):
    return module.apply(variables, **kwargs, method=method)

  @functools.partial(jax.pmap, in_axes=(0, None))
  def predict(images, query_embeddings):

    # Embed images:
    feature_map = apply(module.image_embedder, images=images, train=False)
    b, h, w, d = feature_map.shape
    image_features = jnp.reshape(feature_map, (b, h * w, d))

    # Class predictions are ensembled over query embeddings:
    class_predictor = functools.partial(
        apply, module.class_predictor, image_features=image_features)
    query_embeddings_ensemble = jnp.stack(query_embeddings, axis=0)
    outputs_ensemble = jax.vmap(class_predictor)(
        query_embeddings=query_embeddings_ensemble[:, jnp.newaxis, ...])
    outputs = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0),
                                     outputs_ensemble)

    # Add box predictions:
    outputs.update(
        apply(
            module.box_predictor,
            image_features=image_features,
            feature_map=feature_map))

    outputs[modalities.SCORES] = jax.nn.sigmoid(outputs[modalities.LOGITS])
    return outputs

  return predict


@functools.partial(jax.vmap, in_axes=[0, 0, None, None])  # Map over images.
def get_top_k(
    scores: jnp.ndarray, boxes: jnp.ndarray, k: int,
    exclusive_classes: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Finds the top k scores and corresponding boxes within an image.

  The code applies on the image level; vmap is used for batching.

  Args:
    scores: [num_instances, num_classes] array of scores (i.e. logits or
      probabilities) to sort by.
    boxes: [num_instances, 4] Optional array of bounding boxes.
    k: Number of instances to return.
    exclusive_classes: If True, the top class for each box is returned. If
      False, classes are considered to be non-exclusive (multi-label setting),
      and the top-k computations happens globally across all scores, not just
      the maximum logit for each output token.

  Returns:
    Score, label, and box arrays of shape [top_k, ...] for the selected
    instances.
  """
  if scores.ndim != 2:
    raise ValueError('Expected scores to have shape [num_instances, '
                     f'num_classes], got {scores.shape}')

  if exclusive_classes:
    k = min(k, scores.shape[0])
    instance_top_scores = jnp.max(scores, axis=1)
    instance_class_ind = jnp.argmax(scores, axis=1)
    top_scores, instance_ind = jax.lax.top_k(instance_top_scores, k)
    class_ind = instance_class_ind[instance_ind]
  else:
    k = min(k, scores.size)
    top_scores, top_indices = jax.lax.top_k(scores.ravel(), k)
    instance_ind, class_ind = jnp.unravel_index(top_indices, scores.shape)

  return top_scores, class_ind, boxes[instance_ind]


def unpad_box(box_cxcywh: np.ndarray, *, image_w: int,
              image_h: int) -> np.ndarray:
  """Removes effect of ResizeWithPad-style padding from bounding boxes.

  Args:
    box_cxcywh: Bounding box in COCO format (cx, cy, w, h).
    image_w: Width of the original unpadded image in pixels.
    image_h: Height of the original unpadded image in pixels.

  Returns:
    Unpadded box.
  """
  padded_size = np.maximum(image_w, image_h)
  w_frac = image_w / padded_size
  h_frac = image_h / padded_size
  image_frac = np.array([w_frac, h_frac, w_frac, h_frac]) + 1e-6
  return np.clip(box_cxcywh / image_frac, 0.0, 1.0)


def format_predictions(*,
                       scores: np.ndarray,
                       labels: np.ndarray,
                       boxes: np.ndarray,
                       image_sizes: np.ndarray,
                       image_ids: np.ndarray,
                       label_shift: int = 0) -> List[Dict[str, Any]]:
  """Formats predictions to COCO annotation format.

  Args:
    scores: [num_images, num_instances] array of confidence scores.
    labels: [num_images, num_instances] array of label ids.
    boxes: [num_images, num_instances, 4] array of bounding boxes in relative
      COCO format (cx, cy, w, h).
    image_sizes: [num_images, 2] array of original unpadded image height and
      width in pixels.
    image_ids: COCO/LVIS image IDs.
    label_shift: Value that will be added to the model output labels in the
      prediction JSON files. The model predictions are zero-indexed. COCO or
      LVIS use one-indexed labels, so label_shift should be 1 for these
      datasets. Set it to 0 for zero-indexed datasets.

  Returns:
    List of dicts that can be saved as COCO/LVIS prediction JSON for evaluation.
  """
  predictions = []
  num_batches, num_instances = scores.shape
  for batch in range(num_batches):
    h, w = image_sizes[batch]
    for instance in range(num_instances):
      label = int(labels[batch, instance])
      if not label:
        continue
      score = float(scores[batch, instance])
      # Internally, we use center coordinates, but COCO uses corner coordinates:
      bcx, bcy, bw, bh = unpad_box(boxes[batch, instance], image_w=w, image_h=h)
      bx = bcx - bw / 2
      by = bcy - bh / 2
      predictions.append({
          'image_id': int(image_ids[batch]),
          'category_id': label + label_shift,
          'bbox': [float(bx * w), float(by * h), float(bw * w), float(bh * h)],
          'score': score
      })
  return predictions


def get_predictions(config: ml_collections.ConfigDict,
                    checkpoint_path: Optional[str],
                    tfds_name: str,
                    split: str,
                    top_k: int = 300,
                    exclusive_classes: bool = False,
                    label_shift: int = 0) -> List[Dict[str, Any]]:
  """Gets predictions from an OWL-ViT model for a whole TFDS dataset.

  These predictions can then be evaluated using the COCO/LVIS APIs.

  Args:
    config: Model config.
    checkpoint_path: Checkpoint path (overwrites the path in the model config).
    tfds_name: TFDS dataset to get predictions for.
    split: Dataset split to get predictions for.
    top_k: Number of predictions to retain per image.
    exclusive_classes: If True, the top class for each box is returned. If
      False, classes are considered to be non-exclusive (multi-label setting),
      and the top-k computations happens globally across all scores, not just
      the maximum logit for each output token.
    label_shift: Value that will be added to the model output labels in the
      prediction JSON files. The model predictions are zero-indexed. COCO or
      LVIS use one-indexed labels, so label_shift should be 1 for these
      datasets. Set it to 0 for zero-indexed datasets.

  Returns:
    Dictionary of predictions.
  """

  # Load model and variables:
  module = models.TextZeroShotDetectionModule(
      body_configs=config.model.body,
      normalize=config.model.normalize,
      box_bias=config.model.box_bias)
  module.tokenize('')  # Warm up the tokenizer.
  variables = module.load_variables(checkpoint_path=checkpoint_path)
  embed_queries = get_embed_queries_fn(module, variables)
  predict = get_predict_fn(module, variables)
  pmapped_top_k = jax.pmap(get_top_k, static_broadcasted_argnums=(2, 3))

  # Create dataset:
  dataset, class_names = get_dataset(
      tfds_name=tfds_name,
      split=split,
      input_size=config.dataset_configs.input_size,
      tfds_data_dir=FLAGS.tfds_data_dir,
      tfds_download_dir=FLAGS.tfds_download_dir,
      data_format=FLAGS.data_format)

  # Embed queries:
  query_embeddings = []
  for template in label_ops.CLIP_BEST_PROMPT_TEMPLATES:
    tokenized_queries = tokenize_queries(
        module.tokenize,
        class_names,
        template,
        max_token_len=config.dataset_configs.max_query_length)
    query_embeddings.append(embed_queries(np.array(tokenized_queries)))  # pytype: disable=wrong-arg-types  # jax-ndarray

  # Prediction loop:
  predictions = []
  for batch in tqdm.tqdm(
      dataset.as_numpy_iterator(),
      desc='Inference progress',
      total=int(dataset.cardinality().numpy())):

    outputs = predict(batch[modalities.IMAGE], query_embeddings)  # pytype: disable=wrong-arg-types  # jax-ndarray

    # Selec top k predictions:
    scores, labels, boxes = pmapped_top_k(
        outputs[modalities.SCORES],
        outputs[modalities.PREDICTED_BOXES],
        top_k,
        exclusive_classes,
    )

    # Move to CPU:
    scores, labels, boxes, image_sizes, image_ids = _unshard_and_get([
        scores, labels, boxes, batch[modalities.ORIGINAL_SIZE],
        batch[modalities.IMAGE_ID]
    ])

    # Append predictions:
    predictions.extend(
        format_predictions(
            scores=scores,
            labels=labels,
            boxes=boxes,
            image_sizes=image_sizes,
            image_ids=image_ids,
            label_shift=label_shift))

  return predictions


def _unshard_and_get(tree):
  tree_cpu = jax.device_get(tree)
  return jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), tree_cpu)


def write_predictions(predictions: List[Dict[str, Any]],
                      output_dir: str, split: str) -> str:
  filepath = os.path.join(output_dir, f'predictions_{split}.json')
  if tf.io.gfile.exists(filepath):
    raise ValueError(f'Output file already exists: {filepath}')
  with tf.io.gfile.GFile(filepath, 'w') as f:
    json.dump(predictions, f, indent=4)
  return filepath


def _download_file(url: str, path: str) -> None:
  """Downloads a file from a URL to a path."""
  logging.info('Downloading %s to %s', url, path)
  with tf.io.gfile.GFile(path, 'wb') as output:
    with urllib.request.urlopen(url) as source:
      loop = tqdm.tqdm(total=int(source.info().get('Content-Length')),
                       ncols=80, unit='iB', unit_scale=True, unit_divisor=1024)
      while True:
        buffer = source.read(8192)
        if not buffer:
          break
        output.write(buffer)
        loop.update(len(buffer))


def _download_annotations(annotations_path: str) -> str:
  """Downloads the appropriate annotations file."""
  filename = os.path.basename(annotations_path)
  if filename == 'lvis_v1_val.json':
    tf.io.gfile.makedirs(os.path.dirname(annotations_path))
    zip_path = annotations_path.replace('.json', '.zip')
    _download_file(url=LVIS_VAL_URL, path=zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
      f.extractall(os.path.dirname(annotations_path))
    tf.io.gfile.remove(zip_path)
  else:
    raise ValueError(f'Unknown annotations file: {filename}')

  return annotations_path


def run_evaluation(annotations_path: str,
                   predictions_path: str,
                   data_format: str = 'lvis') -> Dict[str, float]:
  """Runs evaluation and prints metric results."""

  # Copy annotations file in case it's not local:
  with tempfile.TemporaryDirectory() as temp_dir:
    annotations_path_local = os.path.join(
        temp_dir, os.path.basename(annotations_path))
    tf.io.gfile.copy(annotations_path, annotations_path_local)

    if data_format == 'lvis':
      lvis_gt = LVIS(annotations_path_local)
      lvis_dt = LVISResults(lvis_gt, predictions_path)
      lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type='bbox')
      lvis_eval.evaluate()
      lvis_eval.accumulate()
      lvis_eval.summarize()
      lvis_eval.print_results()
      return lvis_eval.results
    elif data_format == 'coco':
      coco_gt = COCO(annotations_path_local)
      coco_dt = coco_gt.loadRes(predictions_path)
      coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      return {k: v for k, v in zip(COCO_METRIC_NAMES, coco_eval.stats)}
    else:
      raise ValueError(f'Unknown data format: {data_format}')


def _set_host_device_count(n):
  xla_flags = os.getenv('XLA_FLAGS', '')
  xla_flags = re.sub(r'--xla_force_host_platform_device_count=\S+', '',
                     xla_flags).split()
  os.environ['XLA_FLAGS'] = ' '.join(
      ['--xla_force_host_platform_device_count={}'.format(n)] + xla_flags)


def plot_box(ax,
             ann,
             color,
             label=True,
             alpha=1.0,
             pad=3,
             labels=None,
             score=None):
  """Plots a single bounding box into axes."""
  x, y, w, h = ann['bbox']
  ax.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y],
          color=color,
          alpha=alpha)
  if label:
    s = str(ann['category_id'])
    if labels is not None and ann['category_id'] in labels:
      s = f"{ann['category_id']}: {labels[ann['category_id']]}"
    if score is not None:
      s = s + ' ' + f'{score:1.2f}'[1:]
    ax.text(
        x + pad,
        y + pad,
        s,
        ha='left',
        va='top',
        color=color,
        fontsize=10,
        fontweight='bold',
        alpha=alpha)


def plot_image(pixels, image_id, gt_by_image, pred_by_image, labels):
  """Plots an image with annotations."""
  fig, axs = plt.subplots(1, 2, figsize=(12, 6))

  # Plot ground-truth:
  ax = axs[0]
  ax.imshow(pixels)
  for ann in gt_by_image[image_id]:
    plot_box(ax, ann, color='g', labels=labels)
  ax.set_title(f'Ground truth (Image ID: {image_id})')

  # Plot prediction:
  ax = axs[1]
  ax.imshow(pixels)
  anns = pred_by_image[image_id]
  if anns:
    n = _MIN_BOXES_TO_PLOT + len(gt_by_image[image_id]) *  _PRED_BOX_PLOT_FACTOR
    n = min(n, len(anns))
    threshold = np.partition(np.array([a['score'] for a in anns]), -n)[-n]
    threshold = max(threshold, _PLOTTING_SCORE_THRESHOLD)
    for ann in gt_by_image[image_id]:
      plot_box(ax, ann, color='g', label=False)
    for ann in anns:
      if ann['score'] <= threshold:
        continue
      plot_box(ax, ann, color='r', labels=labels, score=ann['score'])
  ax.set_title('Predictions')

  fig.tight_layout()
  return fig


def save_examples_images(*, ground_truth_path, pred_path, tfds_name, split,
                         output_dir, num_images, tfds_data_dir):
  """Saves example images to disk."""
  # Prepare annotations:
  with tf.io.gfile.GFile(ground_truth_path, 'r') as f:
    ground_truth = json.load(f)

  with tf.io.gfile.GFile(pred_path, 'r') as f:
    preds = json.load(f)

  gt_by_image = collections.defaultdict(list)
  for gt in ground_truth['annotations']:
    gt_by_image[gt['image_id']].append(gt)

  pred_by_image = collections.defaultdict(list)
  for pred in preds:
    pred_by_image[pred['image_id']].append(pred)

  labels = {cat['id']: cat['name'] for cat in ground_truth['categories']}

  images = list(
      tfds.load(
          tfds_name, split=split,
          data_dir=tfds_data_dir).take(num_images).as_numpy_iterator())

  # Plot and save images:
  file_names = []
  for image in images:
    image_id = image['image/id']
    fig = plot_image(image['image'], image_id, gt_by_image, pred_by_image,
                     labels)
    file_name = f'{image_id}.png'
    file_path = os.path.join(output_dir, file_name)
    with tf.io.gfile.GFile(file_path, 'wb') as f:
      fig.savefig(f, bbox_inches='tight')
    file_names.append(file_name)

  # Save index.html:
  with tf.io.gfile.GFile(os.path.join(output_dir, 'index.html'), 'w') as f:
    f.write('\n'.join([f'<img src="{n}" alt="{n}">' for n in file_names]))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Starting evaluation.')

  # Make CPU cores visible as JAX devices:
  jax.config.update('jax_platform_name', FLAGS.platform)
  if FLAGS.platform == 'cpu':
    _set_host_device_count(max(1, multiprocessing.cpu_count() - 2))

  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  logging.info('JAX devices: %s', jax.device_count())

  # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  compilation_cache.set_cache_dir('/tmp/jax_compilation_cache')

  config_name = os.path.splitext(os.path.basename(FLAGS.config))[0]

  if tf.io.gfile.exists(FLAGS.annotations_path):
    annotations_path = FLAGS.annotations_path
  else:
    annotations_path = _download_annotations(FLAGS.annotations_path)

  predictions = get_predictions(
      config=getattr(configs, FLAGS.config).get_config(),
      checkpoint_path=FLAGS.checkpoint_path,
      tfds_name=FLAGS.tfds_name,
      split=FLAGS.split,
      label_shift=FLAGS.label_shift)

  output_dir = os.path.join(
      FLAGS.output_dir, config_name, FLAGS.tfds_name, _timestamp()
  )
  logging.info('Writing predictions to %s', output_dir)
  tf.io.gfile.makedirs(output_dir)
  predictions_path = write_predictions(predictions, output_dir, FLAGS.split)

  logging.info('Running evaluation...')
  try:
    results = run_evaluation(annotations_path, predictions_path,
                             FLAGS.data_format)
  except IndexError as e:
    logging.exception('IndexError while computing metric.')
    results = {'ERROR': str(e)}

  with tf.io.gfile.GFile(
      os.path.join(output_dir, f'results_{FLAGS.split}.json'), 'w') as f:
    json.dump(results, f, indent=4)

  if FLAGS.num_example_images_to_save:
    logging.info('Saving example images...')
    examples_dir = os.path.join(output_dir, 'examples')
    tf.io.gfile.makedirs(examples_dir)
    save_examples_images(
        ground_truth_path=annotations_path,
        pred_path=predictions_path,
        tfds_name=FLAGS.tfds_name,
        split=FLAGS.split,
        output_dir=examples_dir,
        num_images=FLAGS.num_example_images_to_save,
        tfds_data_dir=FLAGS.tfds_data_dir)

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
