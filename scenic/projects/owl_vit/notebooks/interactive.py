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

"""Functions for the interactive parts of OWL-ViT notebooks."""

import base64
import functools
import json
from typing import Any, Mapping, Union

from bokeh import models
from bokeh.models import callbacks
from bokeh.models import widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scenic.model_lib.base_models import box_utils
from scenic.projects.owl_vit.notebooks import inference

TEXT_INPUT_PY_CALLBACK_NAME = 'text_input_py_callback'
IMAGE_CONDITIONING_PY_CALLBACK_NAME = 'image_conditioning_py_callback'

IMAGE_COND_NMS_IOU_THRESHOLD = 0.7
IMAGE_COND_MIN_CONF = 0.5


def register_text_input_callback(model: inference.Model, image: np.ndarray,
                                 colab_output: Any):
  """Creates and registers the Python part of the text input callback.

  Args:
    model: inference.Model instance.
    image: Uint8 image on which detection will be performed.
    colab_output: google.colab.output module which needs to be imported in the
      Colab notebook with `from google.colab import output as colab_output`.
  """
  callback = functools.partial(
      _text_input_py_callback, model=model, image=image)
  colab_output.register_callback(TEXT_INPUT_PY_CALLBACK_NAME, callback)


def register_box_selection_callback(model: inference.Model,
                                    query_image: np.ndarray,
                                    target_image: np.ndarray,
                                    colab_output: Any) -> None:
  """Creates and registers the Python part of the box selection callback.

  Args:
    model: inference.Model instance.
    query_image: Uint8 image containing the example object.
    target_image: Uint8 image in which similar objects should be detected.
    colab_output: google.colab.output module which needs to be imported in the
      Colab notebook with `from google.colab import output as colab_output`.
  """
  callback = functools.partial(
      _image_conditioning_py_callback, model=model, query_image=query_image,
      target_image=target_image)
  colab_output.register_callback(IMAGE_CONDITIONING_PY_CALLBACK_NAME, callback)


def _text_input_py_callback(comma_separated_queries: str, *,
                            model: inference.Model,
                            image: np.ndarray) -> str:
  """Gets scores for queries and returns updated box colors.

  This callback is called from JavaScript when the query string in the plot's
  text input box is changed.

  All keyword-arguments must be supplied by functools.partial before using this
  function as input to google.colab.kernel.invokeFunction in JavaScript.

  Args:
    comma_separated_queries: Content of the text input box. Should contain
      comma-separated text queries.
    model: Wrapper object for an OWL-ViT model.
    image: Single uint8 Numpy image to perform detection on.

  Returns:
    JSON-encoded color_updates structure that will be used by the Bokeh
    JavaScript to update box colors.
  """
  queries = [q.strip() for q in comma_separated_queries.split(',')]
  queries = tuple(q for q in queries if q)
  num_queries = len(queries)

  if not num_queries:
    return json.dumps({'color_updates': [], 'legend_text_b64': ''})

  # Compute box display alphas based on prediction scores:
  query_embeddings = model.embed_text_queries(queries)
  top_query_ind, scores = model.get_scores(image, query_embeddings, num_queries)
  alphas = np.zeros_like(scores)
  for i in range(num_queries):
    # Select scores for boxes matching the current query:
    query_mask = top_query_ind == i
    if not np.any(query_mask):
      continue
    query_scores = scores[query_mask]

    # Box alpha is scaled such that the best box for a query has alpha 1.0 and
    # the worst box for which this query is still the top query has alpha 0.1.
    # All other boxes will either belong to a different query, or will not be
    # shown.
    max_score = np.max(query_scores) + 1e-6
    query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
    query_alphas = np.clip(query_alphas, 0.0, 1.0)
    alphas[query_mask] = query_alphas

  # Construct color_updates structure for Bokeh:
  color_updates = []
  for i, (query_ind, alpha) in enumerate(zip(top_query_ind, alphas)):
    color_updates.append((i, _get_query_color(query_ind, float(alpha))))

  # Construct new legend:
  legend_text = _get_query_legend_html(queries)
  # Base64-encode legend text so we don't have to deal with HTML/JSON escaping:
  legend_text_b64 = base64.b64encode(legend_text.encode('utf8')).decode('utf8')

  return json.dumps({
      'color_updates': color_updates,
      'legend_text_b64': legend_text_b64,
  })


def get_text_input_js_callback(data_source: models.ColumnDataSource,
                               legend: widgets.Div) -> callbacks.CustomJS:
  """Creates the CustomJS callback that will be triggered upon query entry.

  The JavaScript callback in turn calls back to Python, specifically to a
  function with the name specified by TEXT_INPUT_PY_CALLBACK_NAME, via
  google.colab.kernel.invokeFunction. The Python callback should return a
  color_updates data structure consisting of a list of (box_index, hex_color)
  tuples.

  Args:
    data_source: Bokeh ColumnDataSource linked to bounding boxes. Must contain a
      field called "colors".
    legend: Div widget that will contain the query legend.

  Returns:
    Bokeh CustomJS callback object.
  """
  return callbacks.CustomJS(
      args=dict(data_source=data_source, legend=legend),
      code="""
          (async function() {
            const input = cb_obj.value_input.replace(/[^a-zA-Z0-9 ,_-]/g, '');
            const result = await google.colab.kernel.invokeFunction(
              '""" + TEXT_INPUT_PY_CALLBACK_NAME + """', [input], {});
            var result_text = result['data']['text/plain']
            // Need to strip enclosing quotes since invokeFunction returns str:
            result_text = result_text.substring(1, result_text.length-1);
            const result_json = JSON.parse(result_text);

            const color_updates = result_json['color_updates']
            const colors = data_source.data['colors']
            for (let i = 0; i < color_updates.length; i++) {
                  // Element 0 is the box index, element 1 is the hex color:
                  colors[color_updates[i][0]] = color_updates[i][1]
              }
            data_source.change.emit();
            legend.text = atob(result_json['legend_text_b64'])
          })();
          """)


def _image_conditioning_py_callback(
    geometry_dict: Mapping[str, Union[float, str]],
    *,
    model: inference.Model,
    query_image: np.ndarray,
    target_image: np.ndarray,
) -> str:
  """Updates image conditioning predictions when box is drawn."""
  # Note: Bokeh's y coords are swapped compared to TensorFlow:
  box = (geometry_dict['y1'], geometry_dict['x0'], geometry_dict['y0'],
         geometry_dict['x1'])
  query_embedding, best_box_ind = model.embed_image_query(query_image, box)
  _, _, query_image_boxes = model.embed_image(query_image)

  # TODO(mjlm): Implement multi-query image-conditioned detection.
  num_queries = 1
  top_query_ind, scores = model.get_scores(
      target_image, query_embedding[None, ...], num_queries=1)

  # Apply non-maximum suppression:
  if IMAGE_COND_NMS_IOU_THRESHOLD < 1.0:
    _, _, target_image_boxes = model.embed_image(target_image)
    target_boxes_yxyx = box_utils.box_cxcywh_to_yxyx(target_image_boxes, np)
    for i in np.argsort(-scores):
      if not scores[i]:
        # This box is already suppressed, continue:
        continue
      ious = box_utils.box_iou(
          target_boxes_yxyx[None, [i], :],
          target_boxes_yxyx[None, :, :],
          np_backbone=np)[0][0, 0]
      ious[i] = -1.0  # Mask self-IoU.
      scores[ious > IMAGE_COND_NMS_IOU_THRESHOLD] = 0.0

  # Compute box display alphas based on prediction scores:
  alphas = np.zeros_like(scores)
  for i in range(num_queries):
    # Select scores for boxes matching the current query:
    query_mask = top_query_ind == i
    query_scores = scores[query_mask]
    if not query_scores.size:
      continue

    # Box alpha is scaled such that the best box for a query has alpha 1.0 and
    # the worst box for which this query is still the top query has alpha 0.1.
    # All other boxes will either belong to a different query, or will not be
    # shown.
    max_score = np.max(query_scores) + 1e-6
    query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
    query_alphas[query_alphas < IMAGE_COND_MIN_CONF] = 0.0
    query_alphas = np.clip(query_alphas, 0.0, 1.0)
    alphas[query_mask] = query_alphas

  # Construct color_updates structure for Bokeh:
  color_updates = []
  for i, (query_ind, alpha) in enumerate(zip(top_query_ind, alphas)):
    color_updates.append((i, _get_query_color(query_ind, float(alpha))))

  cx, cy, w, h = (float(c) for c in query_image_boxes[best_box_ind])
  selected_box = {'x': cx, 'y': cy, 'w': w, 'h': h}

  return json.dumps({
      'color_updates': color_updates,
      'selected_box': selected_box,
  })


def get_image_conditioning_js_callback(
    *,
    user_query_box_data_source: models.ColumnDataSource,
    model_query_box_data_source: models.ColumnDataSource,
    pred_box_data_source: models.ColumnDataSource,
) -> callbacks.CustomJS:
  """Creates the CustomJS callback that will be triggered upon query entry.

  The JavaScript callback in turn calls back to Python, specifically to a
  function with the name specified in py_callback_name, via
  google.colab.kernel.invokeFunction. The Python callback should return a
  color_updates data structure consisting of a list of (box_index, hex_color)
  tuples.

  Args:
    user_query_box_data_source: Bokeh ColumnDataSource linked to the query box
      drawn by the user.
    model_query_box_data_source: Bokeh ColumnDataSource linked to the query box
      selected from the model predictions on the source image.
    pred_box_data_source: Bokeh ColumnDataSource linked to the boxes predicted
      for the target image.

  Returns:
    Bokeh CustomJS callback object.
  """
  return callbacks.CustomJS(
      args=dict(
          user_query_box_data_source=user_query_box_data_source,
          model_query_box_data_source=model_query_box_data_source,
          pred_box_data_source=pred_box_data_source),
      code="""
          (async function() {
          // Get query box coordinates:
          const geometry = cb_obj['geometry'];
          const width = geometry['x1'] - geometry['x0'];
          const height = geometry['y0'] - geometry['y1'];
          const x = geometry['x0'] + width/2;
          const y = geometry['y1'] + height/2;

          // Update source image with user-drawn query box:
          const data = user_query_box_data_source.data;
          data['x'][0] = x;
          data['y'][0] = y;
          data['width'][0] = width;
          data['height'][0] = height;
          user_query_box_data_source.change.emit();

          // Update target plot:
          const result = await google.colab.kernel.invokeFunction(
            '""" + IMAGE_CONDITIONING_PY_CALLBACK_NAME + """', [geometry], {});
          var result_text = result['data']['text/plain']
          // Need to strip enclosing quotes since invokeFunction returns str:
          result_text = result_text.substring(1, result_text.length-1);
          const result_json = JSON.parse(result_text);

          const color_updates = result_json['color_updates']
          const colors = pred_box_data_source.data['colors']
          for (let i = 0; i < color_updates.length; i++) {
                // Element 0 is the box index, element 1 is the hex color:
                colors[color_updates[i][0]] = color_updates[i][1]
            }
          pred_box_data_source.change.emit();

          // Update source image with actual query box from model:
          const data2 = model_query_box_data_source.data;
          data2['x'][0] = result_json['selected_box']['x'];
          data2['y'][0] = result_json['selected_box']['y'];
          data2['width'][0] = result_json['selected_box']['w'];
          data2['height'][0] = result_json['selected_box']['h'];
          model_query_box_data_source.change.emit();

          })();
          """)


@functools.lru_cache(maxsize=None)
def _get_query_color(query_ind, alpha=1.0):
  color = plt.get_cmap('Set1')(np.linspace(0, 1, 10))[query_ind % 10, :3]
  color -= np.min(color)
  color /= np.max(color)
  return mpl.colors.to_hex((color[0], color[1], color[2], alpha),
                           keep_alpha=alpha < 1.0)


def _get_query_legend_html(queries):
  html = []
  for i, query in enumerate(queries):
    color = _get_query_color(i)
    html.append(
        f'<span style="color: {color}; font-size: 14pt; font-weight: bold;">'
        f'{query}'
        '</span>')
  return 'Queries: ' + ', '.join(html)
