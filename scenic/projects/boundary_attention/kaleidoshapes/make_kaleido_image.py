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

"""Helper functions for shape generation."""

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw  # pylint: disable=g-multiple-import
from shapely import geometry

MultiPoint = geometry.MultiPoint
Polygon = geometry.Polygon


def choose_valid_point(key, array):
  """Randomly select a non-zero element from a 2D mask.

  Args:
    key: random key
    array: 2D array of non-zero elements

  Returns:
    point: [x,y] ([column,row]) coordinates of non-zero element
  """

  nonzero_indices = np.argwhere(array > 0)
  if np.size(nonzero_indices) == 0:
    point = np.array([])
  else:
    _, subkey = random.split(key)
    ind = random.choice(subkey, jnp.arange(len(nonzero_indices)))
    point = np.flip(np.array(tuple(nonzero_indices[ind])))

  return point


def disk_to_zero(array, center, radius):
  """Set to zero all elements of a 2D array that are WITHIN radius of a point.

  Args:
    array: 2D array of non-zero elements
    center: [x,y] ([column,row]) coordinates of center point
    radius: radius of disk

  Returns:
    array: 2D array of non-zero elements with all elements within radius of
    center
  """

  h, w = array.shape
  x, y = np.meshgrid(np.arange(w), np.arange(h))

  # return array with all elements close to the center replaced by 0
  return np.where((center[0] - x)**2 + (center[1] - y)**2 > radius**2, array, 0)


def inverted_disk_to_zero(array, center, radius):
  """Set to zero all elements of a 2D array that are BEYOND radius of a point.

  Args:
    array: 2D array of non-zero elements
    center: [x,y] ([column,row]) coordinates of center point
    radius: radius of disk

  Returns:
    array: 2D array of non-zero elements with all elements far from center
  """

  h, w = array.shape
  x, y = np.meshgrid(np.arange(w), np.arange(h))

  # return array with all elements far from center replaced by 0
  return np.where((center[0] - x)**2 + (center[1] - y)**2 < radius**2, array, 0)


def boundary_to_zero(array, dist):
  """Set to zero all elements within dist of the outer boundary.

  Args:
    array: 2D array of non-zero elements
    dist: distance of boundary

  Returns:
    array: 2D array of non-zero elements with all elements within dist of
    boundary
  """

  dist = max(dist, 1)
  array[:int(dist), :] = 0   # top rows
  array[-int(dist):, :] = 0  # bottom rows
  array[:, :int(dist)] = 0   # left columns
  array[:, -int(dist):] = 0  # right columns

  return array


def linesegment_to_zero(array, endpoints, dist):
  """Set to zero all elements of a 2D array...

  ...that are WITHIN perpendicular-distance D of the join of two points.

  Args:
    array: 2D array of non-zero elements
    endpoints: [x,y] ([column,row]) coordinates of endpoints
    dist: distance of line

  Returns:
    array: 2D array of non-zero elements with all elements within
    perpendicular-distance D of line
  """

  h, w = array.shape
  x, y = np.meshgrid(np.arange(w), np.arange(h))

  # compute line (a,b,c) from endpoints
  line = [endpoints[0][1] - endpoints[1][1],
          endpoints[1][0] - endpoints[0][0],
          endpoints[0][0]*endpoints[1][1] - endpoints[1][0]*endpoints[0][1]]

  # return array with all elements close to the line replaced by 0
  return np.where(np.abs(
      line[0]*x + line[1]*y + line[2]) / np.sqrt(
          line[0]**2 + line[1]**2) > dist, array, 0)


def inverted_linesegment_to_zero(array, endpoints, dist):
  """Set to zero all elements of a 2D array that are BEYOND...

  ...perpendicular-distance D of the join of two points.

  Args:
    array: 2D array of non-zero elements
    endpoints: [x,y] ([column,row]) coordinates of endpoints
    dist: distance of line

  Returns:
    array: 2D array of non-zero elements with all elements far from line
  """

  h, w = array.shape
  x, y = np.meshgrid(np.arange(w), np.arange(h))

  # compute line (a,b,c) from endpoints
  line = [endpoints[0][1] - endpoints[1][1],
          endpoints[1][0] - endpoints[0][0],
          endpoints[0][0]*endpoints[1][1] - endpoints[1][0]*endpoints[0][1]]

  # return array with all elements close to the line replaced by 0
  return np.where(np.abs(
      line[0]*x + line[1]*y + line[2]) / np.sqrt(
          line[0]**2 + line[1]**2) < dist, array, 0)


def sample_random_color(key):
  """Generate a random RGB vector (in [0,255]).

  Args:
    key: random key

  Returns:
    tuple: RGB color
  """

  rgb = random.randint(key, (3,), minval=0, maxval=256)
  return tuple(rgb.tolist())


def sample_triangle(key, config, h=240, w=320):
  """Randomly sample a triangle with constraints.

  Args:
    key: random key
    config: configuration object (hyper-parameters for shape generation)
    h: height of image
    w: width of image

  Returns:
    triangle vertices [(x1,y1), (x2,y2), (x3,y3)]
  """

  # initialize vertex lists
  vertices = []

  # Note: Our strategy for handling shape constraints is to randomly sample
  # integer values for the vertex locations over the grid [0,w-1]x[0,h-1]
  # (which allows imposing arbitrary constraints via a pixelated binary mask)
  # and then add a random sub-pixel displacement from Uniform([-0.5,0.5])

  # create a binary mask to kep track of valid choices of vertex locations
  vertex_options = np.ones((h, w), dtype=np.uint8)

  # first vertex can be anywhere except within boundary_width*max(w,h) of the
  # image boundary
  vertex_options = boundary_to_zero(vertex_options,
                                    config.boundary_width * max(w, h))
  next_vertex = choose_valid_point(key, vertex_options)
  if np.size(next_vertex) == 0:
    raise ValueError('Triangle sampling: cannot find valid vertex')
  else:
    key, subkey = random.split(key)
    next_vertex = next_vertex + random.uniform(subkey, shape=(2,),
                                               minval=-0.5, maxval=0.5)
    vertices.append(next_vertex)

  # second vertex must additionally be in a donut around the first one, as
  # determined by [min_triangle_base, max_triangle_base]*max(w,h)
  vertex_options = disk_to_zero(vertex_options, vertices[0],
                                config.min_triangle_base * max(w, h))
  vertex_options = inverted_disk_to_zero(vertex_options, vertices[0],
                                         config.max_triangle_base * max(w, h))
  next_vertex = choose_valid_point(key, vertex_options)
  if np.size(next_vertex) == 0:
    raise ValueError('Triangle sampling: cannot find valid vertex')
  else:
    key, subkey = random.split(key)
    next_vertex = next_vertex + random.uniform(subkey,
                                               shape=(2,),
                                               minval=-0.5,
                                               maxval=0.5)
    vertices.append(next_vertex)

  # third vertex must additionally be at least [min_triangle_height*max(w,h)
  # from the line connecting the first two vertices
  vertex_options = linesegment_to_zero(vertex_options, vertices,
                                       config.min_triangle_height * max(w, h))
  next_vertex = choose_valid_point(key, vertex_options)
  if np.size(next_vertex) == 0:
    raise ValueError('Triangle sampling: cannot find valid vertex')
  else:
    _, subkey = random.split(key)
    next_vertex = next_vertex + random.uniform(subkey, shape=(2,),
                                               minval=-0.5, maxval=0.5)
    vertices.append(next_vertex)

  # compute from pixel units to normalized units, and from arrays to tuples
  vertices = [tuple(v / max(w, h)) for v in vertices]

  return vertices


def render_image_from_shapes(shapes, shapecolors, basecolor, h=240, w=320):
  """Render image using dictionary of shapes and associated RGB colors.

  Args:
    shapes: list of shape dictionaries, in order of back-object to front-object
    shapecolors: list of RGB color values (in [0,255])
    basecolor: RBG color of background (in [0,255])
    h: height of image
    w: width of image

  Returns:
    image:      RGB image of shape (h, w, 3)
    boundaries: binary image of pixelated boundariues (h, w, 1)
    segments:   integer-value image indicating shape segmentations
    (0=background, 1=shapes[0], 2=shapes[1],...)

  Shape dictionary:
    'type' : {'circle', 'triangle'}
    'params' : {(cx,cy,r), [(x1,y1), (x2,y2), (x3,y3)]}
  """

  # create blank image with base_color
  im = Image.new('RGB', (w, h), basecolor)
  draw_im = ImageDraw.Draw(im)

  # create binary all-zeros image for (pixelated) boundaries
  im_b = Image.new('1', (w, h), 0)
  draw_b = ImageDraw.Draw(im_b)

  # create single-channel (grayscale) all-zeros image for segment indicators
  im_s = Image.new('L', (w, h), color=0)
  draw_s = ImageDraw.Draw(im_s)

  # iterate through list from, drawing color-filled shapes and white-filled
  #   shapes using back-to-front depth ordering
  for i, shape in enumerate(shapes):
    if shape['type'] == 'circle':

      # draw circle
      circle_params = np.array(shape['params'])  # (x,y,r)
      circle_params = circle_params * max(w, h)

      # convert from (cx,cy,r) to bounding box (x1,y1,x2,y2)
      circle_bbox = (circle_params[0]-circle_params[2],
                     circle_params[1]-circle_params[2],
                     circle_params[0]+circle_params[2],
                     circle_params[1]+circle_params[2])

      draw_im.ellipse(circle_bbox, fill=shapecolors[i])
      draw_b.ellipse(circle_bbox, fill=0, outline=1, width=1)
      draw_s.ellipse(circle_bbox, fill=i+1)

    elif shape['type'] == 'triangle':

      # draw triangle
      verts = np.array(shape['params'])  # [(x1,y1), (x2,y2), (x3,y3)]
      verts = verts * max(w, h)
      verts = verts.ravel().tolist()  # make into a list for PIL

      draw_im.polygon(verts, fill=shapecolors[i])
      draw_b.polygon(verts, fill=0, outline=1, width=1)
      draw_s.polygon(verts, fill=i+1)

  # render bitmaps
  image = np.array(im.convert())
  boundaries = np.array(im_b)
  segments = np.array(im_s.convert('L'))

  return image, boundaries, segments


def compute_distance_from_shapes(shapes, h=240, w=320):
  """Compute distance map using dictionary of shapes.

  Args:
    shapes: list of shape dictionaries, in order of back-object to front-object
    h: height of image
    w: width of image

  Returns:
    dmap: jax array of size (h, w

  Shape dictionary:
  'type' : {'circle', 'triangle'}
  'params' : {(cx,cy,r), [(x1,y1), (x2,y2), (x3,y3)]}
  """

  # initialize distance map
  dmap = np.full((h, w), np.inf)
  dmap = dmap.ravel()

  # Create 2D coordinate arrays
  x, y = np.meshgrid(np.arange(w), np.arange(h))
  x = x.ravel()
  y = y.ravel()

  # Create Shapely MultiPoint object from coordinate arrays
  points = MultiPoint(np.column_stack((x.ravel(), y.ravel())))

  # iterate through list of shapes from back to front
  for _, shape in enumerate(shapes):
    if shape['type'] == 'circle':

      # get circle parameters and scale to image size
      circle_params = np.array(shape['params'])  # (x,y,r)
      circle_params = circle_params * max(w, h)

      # signed distance from circle
      shapedist = np.sqrt(
          (x - circle_params[0]) ** 2 + (
              y - circle_params[1]) ** 2) - circle_params[2]

      # foreground mask
      shapemask = shapedist <= 0

      # unsigned distance from disk
      shapedist = np.abs(shapedist)

      # Composite with previous distance map
      dmap = np.where(shapemask, shapedist, np.minimum(shapedist, dmap))

    elif shape['type'] == 'triangle':

      # get triangle parameters and scale to image size
      verts = np.array(shape['params'])
      verts = verts * max(w, h)
      verts = verts.tolist()

      # create Shapely Polygon object for the triangle
      poly = Polygon(verts)

      # unsigned distance from polygon's outline
      shapedist = np.array([p.distance(poly.boundary) for p in points.geoms])

      # foreground mask
      shapemask = np.array([p.distance(poly) for p in points.geoms]) == 0

      # Composite with previous distance map
      dmap = np.where(shapemask, shapedist, np.minimum(shapedist, dmap))

  return dmap.reshape(h, w)


def sample_shapes(key, config, h=240, w=320):
  """Randomly sample a set of shapes.

  Args:
    key: random key
    config: configuration object (hyper-parameters for shape generation)
    h: height of image
    w: width of image

  Returns:
    List of shape dictionaries, in order of back-object to front-object

  Shape dictionary:
    'type' : {'circle', 'triangle'}
    'params' : {(x,y,r), [(x1,y1), (x2,y2), (x3,y3)]}
  """

  aspect_ratio = (w/max(w, h), h/max(w, h))

  shapes = []

  key, subkey = random.split(key)
  for _ in range(jax.random.randint(subkey, (1,), config.min_objects,
                                    config.max_objects)[0]):

    # choose circle or triangle
    key, subkey = random.split(key)
    if random.bernoulli(subkey, config.prob_circle):

      # make circle
      key, *subkeys = random.split(key, 4)
      radius = random.uniform(subkeys[0],
                              minval=config.min_radius,
                              maxval=config.max_radius)
      center_x = random.uniform(
          subkeys[1],
          minval=(radius + config.boundary_width),
          maxval=(aspect_ratio[0] - radius - config.boundary_width))
      center_y = random.uniform(
          subkeys[2],
          minval=(radius + config.boundary_width),
          maxval=(aspect_ratio[1] - radius - config.boundary_width))
      circparams = jnp.stack([center_x, center_y, radius]).tolist()

      shape = {'type': 'circle',
               'params': circparams}

    else:

      # make triangle
      key, subkey = random.split(key)
      verts = sample_triangle(subkey, config, h=h, w=w)

      shape = {'type': 'triangle',
               'params': verts}

    shapes.append(shape)

  return shapes


def filter_shape_image(config, imagedict):
  """Remove shapes that are not visible in the image.

  Args:
    config: configuration object (hyper-parameters for shape generation)
    imagedict: shape_image dictionary

  Returns:
    filtered_shape_image: shape_image dictionary

  Notes:
    imagedict dictionary:
      'height': height
      'width': width
      'num_shapes': number of shapes
      'shapes': list of shape dictionaries (see below)
      'shapecolors': list of RGB shapecolors
      'basecolor': background color
      'image': RGB image
      'boundaries': boundary image
      'segments': segmentation map
      'distance': distance map

    shape dictionary:
      'type' : {'circle', 'triangle'}
      'params' : {(x,y,r), [(x1,y1), (x2,y2), (x3,y3)]}
  """

  # number of input shapes
  num_input_shapes = imagedict['num_shapes']

  # scaled visbility threshold in proportion to image size
  vis_threshold = config.min_visibility * (
      imagedict['width'] * imagedict['height'])

  num_shapes_orig = len(imagedict['shapes'])

  imagedict['shapes'] = [e for i, e in enumerate(
      imagedict['shapes']) if (np.sum(
          imagedict['segments'] == i+1) > vis_threshold)]
  imagedict['shapecolors'] = [e for i, e in enumerate(
      imagedict['shapecolors']) if (np.sum(
          imagedict['segments'] == i+1) > vis_threshold)]

  # update number of shapes
  imagedict['num_shapes'] = len(imagedict['shapes'])

  # if any shapes have been removed, then re-render image and maps
  if imagedict['num_shapes'] < num_input_shapes:
    image, boundaries, segments = render_image_from_shapes(
        imagedict['shapes'],
        imagedict['shapecolors'],
        imagedict['basecolor'],
        imagedict['height'],
        imagedict['width'])
    imagedict['image'] = image
    imagedict['boundaries'] = boundaries
    imagedict['segments'] = segments

  return imagedict, num_shapes_orig - imagedict['num_shapes']


# Helper function to check if a point is inside a given shape
def point_inside_shape(point, shape):
  if shape['type'] == 'circle':
    return inside_circle(point, np.array(shape['params'][:2]),
                         np.array(shape['params'][2]))
  elif shape['type'] == 'triangle':
    return inside_triangle(point, list(np.array(shape['params'])))


def circle_circle_intersections(circle1, circle2):
  """Find the points where two circles intersect."""

  c1_center, c1_radius = circle1
  c2_center, c2_radius = circle2

  x1, y1 = c1_center
  x2, y2 = c2_center
  r1 = c1_radius
  r2 = c2_radius

  # Distance between the centers
  d = np.linalg.norm(c2_center - c1_center)

  # No solution conditions
  if d > r1 + r2 or d < abs(r1 - r2) or (d == 0 and r1 != r2):
    return []

  # Compute the formula variables
  a = (r1**2 - r2**2 + d**2) / (2 * d)
  h = np.sqrt(r1**2 - a**2)

  # Compute the point P, which is the intersection of the line passing
  # through the circle centers and the line orthogonal to it where the circles
  # intersect
  px = x1 + a * (x2 - x1) / d
  py = y1 + a * (y2 - y1) / d

  # Intersection points
  x3_1 = px + h * (y2 - y1) / d
  y3_1 = py - h * (x2 - x1) / d

  x3_2 = px - h * (y2 - y1) / d
  y3_2 = py + h * (x2 - x1) / d

  if h == 0:  # The circles are tangent
    return [(x3_1, y3_1)]
  else:
    return [(x3_1, y3_1), (x3_2, y3_2)]


def inside_circle(point, center, radius):
  """Check if a point is inside a circle."""
  return np.sum((point - center)**2) < radius**2


def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2):
  """Find the points where the circle intersects with the line segment."""
  intersections = []

  a = (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2
  b = 2 * ((pt2[0] - pt1[0]) * (pt1[0] - circle_center[0]) +
           (pt2[1] - pt1[1]) * (pt1[1] - circle_center[1]))

  c = circle_center[0]**2 + circle_center[1]**2 + pt1[0]**2 + pt1[1]**2 - 2 * (
      circle_center[0] * pt1[0] + circle_center[1] * pt1[1]
      ) - circle_radius**2

  discriminant = b**2 - 4 * a * c

  if discriminant >= 0:
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    if 0 <= t1 <= 1:
      intersections.append((1 - t1) * pt1 + t1 * pt2)

    if 0 <= t2 <= 1:
      intersections.append((1 - t2) * pt1 + t2 * pt2)

  return intersections


def triangle_circle_intersections(triangle, circle):
  """Find the points where the circle intersects with the triangle."""

  circle_center, circle_radius = circle

  intersections = []

  # Check intersection with each edge of the triangle
  for i in range(3):
    edge_start = triangle[i]
    edge_end = triangle[(i + 1) % 3]
    intersections.extend(circle_line_segment_intersection(circle_center,
                                                          circle_radius,
                                                          edge_start, edge_end))

  return intersections


def inside_triangle(pt, triangle):
  """Check if a point pt is inside the triangle."""
  # Convert triangle to barycentric coordinates
  a, b, c = triangle
  v0 = b - a
  v1 = c - a
  v2 = pt - a

  # Compute dot products
  dot00 = np.dot(v0, v0)
  dot01 = np.dot(v0, v1)
  dot02 = np.dot(v0, v2)
  dot11 = np.dot(v1, v1)
  dot12 = np.dot(v1, v2)

  # Compute barycentric coordinates
  inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
  u = (dot11 * dot02 - dot01 * dot12) * inv_denom
  v = (dot00 * dot12 - dot01 * dot02) * inv_denom

  # Check if point is in triangle
  return (u >= 0) and (v >= 0) and (u + v <= 1)


def on_segment(p, q, r):
  """Check if point q lies on line segment pr."""

  return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
          q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))


def orientation(p, q, r):
  """Determine the orientation of the triplet (p, q, r)."""

  val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
  if val == 0: return 0
  return 1 if val > 0 else 2


def line_intersection(line1, line2):
  """Return the intersection point of two lines (if it exists)."""

  xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
  ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

  def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

  div = det(xdiff, ydiff)
  if div == 0:
    return None

  d = (det(*line1), det(*line2))
  x = det(d, xdiff) / div
  y = det(d, ydiff) / div

  return x, y


def triangle_triangle_intersections(triangle1, triangle2):
  """Check if two triangles intersect and return the intersection points."""
  intersections = []
  for i in range(3):
    for j in range(3):
      if do_intersect(triangle1[i], triangle1[(i+1)%3], triangle2[j],
                      triangle2[(j+1)%3]):
        intersection = line_intersection((triangle1[i],
                                          triangle1[(i+1)%3]),
                                         (triangle2[j],
                                          triangle2[(j+1)%3]))

        if intersection and intersection not in intersections:
          intersections.append(intersection)
  return intersections


def do_intersect(p1, q1, p2, q2):
  """Check if the line segments p1q1 and p2q2 intersect."""
  o1 = orientation(p1, q1, p2)
  o2 = orientation(p1, q1, q2)
  o3 = orientation(p2, q2, p1)
  o4 = orientation(p2, q2, q1)

  if o1 != o2 and o3 != o4:
    return True

  if o1 == 0 and on_segment(p1, p2, q1): return True
  if o2 == 0 and on_segment(p1, q2, q1): return True
  if o3 == 0 and on_segment(p2, p1, q2): return True
  if o4 == 0 and on_segment(p2, q1, q2): return True

  return False


def compute_intersections_and_vertices(shapes):
  """Compute the intersections and vertices of a list of shapes."""

  all_intersections = []
  all_vertices = []

  for i in range(len(shapes)):

    shape_i = shapes[i]

    if shape_i['type'] == 'triangle':
      vertices = shape_i['params']

      # Check for occulusions
      occluded = False
      for vertex in vertices:

        for k in range(0, len(shapes)):
          shape_k = shapes[k]

          if point_inside_shape(vertex, shape_k) and ((k > i)):
            occluded = True
            break

        if not occluded:
          all_vertices.append(vertex)

    for j in range(i+1, len(shapes)):

      shape_j = shapes[j]

      intersections = []

      if shape_i['type'] == 'circle':
        if shape_j['type'] == 'circle':
          intersections = circle_circle_intersections(
              (np.array(shape_i['params'][:2]),
               np.array(shape_i['params'][2])),
              (np.array(shape_j['params'][:2]),
               np.array(shape_j['params'][2])))
        elif shape_j['type'] == 'triangle':
          intersections = triangle_circle_intersections(
              np.array(shape_j['params']),
              (np.array(shape_i['params'][:2]),
               np.array(shape_i['params'][2])))
      elif shape_i['type'] == 'triangle':
        if shape_j['type'] == 'triangle':
          intersections = triangle_triangle_intersections(
              np.array(shape_i['params']),
              np.array(shape_j['params']))
        elif shape_j['type'] == 'circle':
          intersections = triangle_circle_intersections(
              np.array(shape_i['params']),
              (np.array(shape_j['params'][:2]),
               np.array(shape_j['params'][2])))

      # Check for occlusions
      if len(intersections) == 0:  # pylint: disable=g-explicit-length-test
        continue

      for intersection in intersections:
        if len(intersection) == 0:  # pylint: disable=g-explicit-length-test
          continue

        occluded = False
        for k in range(0, len(shapes)):

          shape_k = shapes[k]

          if point_inside_shape(intersection,
                                shape_k) and ((k > i) & (k != j)):
            occluded = True
            break

        if not(occluded):
          all_intersections.append(intersection)

  return all_intersections, all_vertices


def generate_image(im_idx, config):
  """Generate a kaleidoshapes image.

  Args:
    im_idx: index of image to generate
    config: config object

  Returns:
    imagedict: dictionary of image data
  """

  # initialize random key with image index as the seed
  key = jax.random.PRNGKey(im_idx)

  # sample a random shape set
  shapes = sample_shapes(key, config, h=config.image_height,
                         w=config.image_width)

  # sample random shape-colors and basecolor
  key, *subkeys = jax.random.split(key, len(shapes)+1)
  shapecolors = [sample_random_color(subkey) for subkey in subkeys]

  _, subkey = jax.random.split(key)
  basecolor = sample_random_color(subkey)

  # make the image, boundary map and segmentation map
  image, boundaries, segments = render_image_from_shapes(shapes,
                                                         shapecolors,
                                                         basecolor,
                                                         h=config.image_height,
                                                         w=config.image_width)
  distance = compute_distance_from_shapes(shapes, h=config.image_height,
                                          w=config.image_width)

  # organize into a dictionary
  imagedict = {
      'height': boundaries.shape[0],
      'width': boundaries.shape[1],
      'num_shapes': len(shapes),
      'shapes': shapes,
      'shapecolors': shapecolors,
      'basecolor': basecolor,
      'image': image,
      'boundaries': boundaries,
      'segments': segments,
      'distance': distance
    }

  imagedict, _ = filter_shape_image(config, imagedict)
  imagedict['distance'] = compute_distance_from_shapes(imagedict['shapes'],
                                                       h=imagedict['height'],
                                                       w=imagedict['width'])

  all_intersections, all_vertices = compute_intersections_and_vertices(
      imagedict['shapes'])
  num_intersections = len(all_intersections)
  num_vertices = len(all_vertices)

  intersections = np.zeros((2700, 2), dtype=np.float32)

  if  num_intersections > 0:
    intersections[:num_intersections] = np.array(all_intersections,
                                                 dtype=np.float32)

  circle_shape_params = np.zeros((config.max_objects, 3), dtype=np.float32)
  triangle_shape_params = np.zeros((config.max_objects, 3, 2), dtype=np.float32)
  shape_colors = np.zeros((config.max_objects, 3), dtype=np.uint8)
  shape_types = ['None']*config.max_objects

  for ii in range(imagedict['num_shapes']):
    shape_types[ii] = imagedict['shapes'][ii]['type']
    shape_colors[ii] = imagedict['shapecolors'][ii]

    if shape_types[ii] == 'circle':
      circle_shape_params[ii] = imagedict['shapes'][ii]['params']
    else:
      triangle_shape_params[ii] = np.array(imagedict['shapes'][ii]['params'])

  vertices = np.zeros((75, 2), dtype=np.float32)

  if all_vertices:
    vertices[:num_vertices, :] = np.array(all_vertices, dtype=np.float32)

  final_imagedict = {'image_index': im_idx,
                     'image': np.array(imagedict['image'], dtype=np.uint8),
                     'boundaries': np.expand_dims(np.array(
                         imagedict['boundaries'], dtype=np.uint8), -1),
                     'segments': np.expand_dims(np.array(
                         imagedict['segments'], dtype=np.uint8), -1),
                     'distances': np.array(imagedict['distance'],
                                           dtype=np.float32),
                     'num_shapes': imagedict['num_shapes'],
                     'shapes': {'type': shape_types,
                                'color': shape_colors,
                                'triangle_params': triangle_shape_params,
                                'circle_params': circle_shape_params},
                     'basecolor': np.array(imagedict['basecolor'],
                                           dtype=np.uint8),
                     'intersections': intersections,
                     'num_intersections': num_intersections,
                     'vertices': vertices,
                     'num_vertices': num_vertices
                    }
  return final_imagedict
