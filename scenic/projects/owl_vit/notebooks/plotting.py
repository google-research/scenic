"""OWL-ViT notebool plotting functions."""
from bokeh import events
from bokeh import layouts
from bokeh import models
from bokeh import plotting
from bokeh.models import widgets
import numpy as np
from scenic.model_lib.base_models import box_utils
from scenic.projects.owl_vit.notebooks import interactive


def create_text_conditional_figure(image: np.ndarray,
                                   boxes: np.ndarray,
                                   fig_size: int = 900) -> layouts.LayoutDOM:
  """Creates a Bokeh figure for interactive text-conditional detection.

  Args:
    image: Image to detect objects in.
    boxes: All predicted boxes for the image, in [cx, cy, w, h] format.
    fig_size: Size of the Bokeh figure in pixels.

  Returns:
    The Bokeh layout of the figure.
  """
  plot = _create_image_figure(image, fig_size)
  box_data_source = _plot_boxes(plot, boxes)

  # Create div that will display the query legend:
  legend = widgets.Div(text='', height=30, width=plot.plot_width - 35)

  # Create text input and register callback:
  text_input = widgets.TextInput(
      value='',
      title='Enter comma-separated queries:',
      width=plot.plot_width - 35)
  text_input.js_on_change(
      'value_input',
      interactive.get_text_input_js_callback(box_data_source, legend))

  # Assemble and show figure:
  layout = layouts.column(text_input, legend, plot)
  plotting.show(layout)

  return layout


def create_image_conditional_figure(query_image: np.ndarray,
                                    target_image: np.ndarray,
                                    target_boxes: np.ndarray,
                                    fig_size: int = 600) -> layouts.LayoutDOM:
  """Creates a Bokeh figure for interactive image-conditional detection.

  Args:
    query_image: Image from which the query box will be selected.
    target_image: Image in which to detect objects.
    target_boxes: Predicted boxes for the target image ([cx, cy, w, h] format).
    fig_size: Size of the Bokeh figure in pixels.

  Returns:
    The Bokeh layout of the figure.
  """
  source_plot = _create_image_figure(
      query_image, fig_size, title='Query image', tools='box_select')
  target_plot = _create_image_figure(
      target_image, fig_size, title='Target image')
  pred_box_data_source = _plot_boxes(target_plot, target_boxes)

  # Source selection code:
  user_query_rect = models.Rect(
      x='x',
      y='y',
      width='width',
      height='height',
      line_color='#00ff00',
      line_width=3,
      fill_alpha=0.1,
      fill_color='#00ff00')
  user_query_box_data_source = models.ColumnDataSource(
      data=dict(x=(-1,), y=(-1,), width=(0,), height=(0,)))
  source_plot.add_glyph(
      user_query_box_data_source,
      user_query_rect,
      selection_glyph=user_query_rect,
      nonselection_glyph=user_query_rect)
  model_query_rect = models.Rect(
      x='x',
      y='y',
      width='width',
      height='height',
      line_color='#ff0000',
      line_width=3,
      fill_alpha=0.1,
      fill_color='#ff0000')
  model_query_box_data_source = models.ColumnDataSource(
      data=dict(x=(-1,), y=(-1,), width=(0,), height=(0,)))
  source_plot.add_glyph(
      model_query_box_data_source,
      model_query_rect,
      selection_glyph=model_query_rect,
      nonselection_glyph=model_query_rect)

  # Register box selection callback:
  callback = interactive.get_image_conditioning_js_callback(
      user_query_box_data_source=user_query_box_data_source,
      model_query_box_data_source=model_query_box_data_source,
      pred_box_data_source=pred_box_data_source)
  source_plot.js_on_event(events.SelectionGeometry, callback)

  layout = layouts.row(source_plot, target_plot)
  plotting.show(layout)

  return layout


def _create_image_figure(image: np.ndarray,
                         fig_size: int = 900,
                         title: str = '',
                         tools: str = '') -> plotting.Figure:
  """Creates a Bokeh figure showing an image."""
  # Determine relative width and height from padding. We assume that padding is
  # added on the bottom or right and has value 0.5:
  width = np.mean(np.any(image[..., 0] != 0.5, axis=0))
  height = np.mean(np.any(image[..., 0] != 0.5, axis=1))
  plot_width = int(width * fig_size)
  plot_height = int(height * fig_size)
  plot = plotting.figure(
      title=title,
      plot_width=plot_width,
      plot_height=plot_height,
      x_range=[0., width],
      y_range=[height, 0.],
      tools=tools)
  plot.axis.visible = False
  plot.toolbar.logo = None
  plot.image_rgba(image=[_bokeh_format_image(image)], x=0., y=1., dw=1., dh=1.)
  return plot


def _plot_boxes(plot: plotting.Figure,
                boxes: np.ndarray,
                line_width: float = 3,
                initial_color: str = '#00000000') -> models.ColumnDataSource:
  """Adds boxes to the provided Bokeh plot."""
  xs = []
  ys = []
  colors = []
  boxes = box_utils.box_cxcywh_to_yxyx(boxes, np)
  for box in boxes:
    y0, x0, y1, x1 = box
    xs.append([x0, x1, x1, x0, x0])
    ys.append([y1, y1, y0, y0, y1])
    colors.append(initial_color)
  box_data_source = models.ColumnDataSource(
      data=dict(xs=xs, ys=ys, colors=colors))
  plot.multi_line(
      source=box_data_source,
      xs='xs',
      ys='ys',
      line_color='colors',
      line_width=line_width)
  return box_data_source


def _bokeh_format_image(image):
  """Formats an RGB image (range [0.0, 1.0], shape [H, W, 3]) for bokeh."""
  # Add alpha layer:
  image = np.concatenate((image, np.ones_like(image[..., :1])), axis=-1)
  image = np.flipud(image) * 255
  return image.astype(np.uint8).view(np.uint32).reshape(image.shape[:2])
