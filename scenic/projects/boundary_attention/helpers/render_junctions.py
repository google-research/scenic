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

"""Renders the junctions of the wedge support and distance maps."""

import jax.numpy as jnp


class JunctionRenderer:
  """Renders the junctions of the wedge support and distance maps."""

  def render_wedges(self,
                    vertex: jnp.ndarray,
                    centralangles: jnp.ndarray,
                    wedgeangles: jnp.ndarray,
                    patchmin: float,
                    patchmax: float,
                    patchres: int,
                    eta: float) -> jnp.ndarray:
    """Render an integer-valued image of the wedge supports over a square patch.

    Args:
      vertex: Array of shape [N, 2, H, W] containing the u and v coordinates
      of vertices
      centralangles: Array of shape [N, 3, H, W] containing the three central
      angles (wedge directions)
      wedgeangles: Array of shape [N, 3, H, W] containing the three wedge angles
      that sum to 2*pi
      patchmin: Minimum value of the patch
      patchmax: Maximum value of the patch
      patchres: Size of the patch in pixels
      eta: Array of shape [N, 1] containing the angular speed of the wedge
      support

    Returns:
      Array of shape [N, M, R, R, H, W]
    """

    # coordinate grid of pixel locations
    yt = jnp.expand_dims(jnp.expand_dims(jnp.linspace(patchmin,
                                                      patchmax,
                                                      patchres),
                                         -1), (0, 1, 4, 5))
    xt = jnp.expand_dims(jnp.expand_dims(jnp.linspace(patchmin,
                                                      patchmax,
                                                      patchres),
                                         0), (0, 1, 4, 5))

    x0 = jnp.expand_dims(vertex[:, 0, ...], (1, 2, 3))
    y0 = jnp.expand_dims(vertex[:, 1, ...], (1, 2, 3))

    cos_ca = jnp.cos(centralangles)
    sin_ca = jnp.sin(centralangles)

    x = ((xt - x0) * cos_ca +
         (yt - y0) * sin_ca  -
         jnp.cos(wedgeangles/2) * jnp.sqrt((xt - x0)**2 + (yt - y0)**2))

    x = 0.5 * (1.0 + (2.0 / jnp.pi) * jnp.arctan(x / eta))
    x = x/jnp.sum(x, axis=1, keepdims=True)

    return x

  def render_distance(self,
                      vertex: jnp.ndarray,
                      boundaryangles: jnp.ndarray,
                      patchmin: float,
                      patchmax: float,
                      patchres: int):
    """Render a distance map over a square patch.

    Args:
      vertex: Array of shape [N, 2, H, W] containing the u and v coordinates
      of the vertices
      boundaryangles: Array of shape [N, 3, H, W] containing the three boundary
      angles (boundary-ray directions)
      patchmin: Minimum value of the patch
      patchmax: Maximum value of the patch
      patchres: Size of the patch

    Returns:
      Array of shape [N, 1, R, R, H, W]
    """

    # coordinate grid of pixel locations
    yt = jnp.expand_dims(jnp.expand_dims(jnp.linspace(patchmin,
                                                      patchmax,
                                                      patchres),
                                         -1),
                         (0, 1, 4, 5))
    xt = jnp.expand_dims(jnp.expand_dims(jnp.linspace(patchmin,
                                                      patchmax,
                                                      patchres),
                                         0),
                         (0, 1, 4, 5))

    x0 = jnp.expand_dims(vertex[:, 0, ...], (1, 2, 3))
    y0 = jnp.expand_dims(vertex[:, 1, ...], (1, 2, 3))

    cos_ba = jnp.cos(boundaryangles)
    sin_ba = jnp.sin(boundaryangles)

    distance_branches = jnp.where(0 < ((xt - x0)*cos_ba + (yt - y0)*sin_ba),
                                  jnp.abs(-(xt - x0)*sin_ba + (yt - y0)*cos_ba),
                                  jnp.sqrt((xt - x0)**2 + (yt - y0)**2))

    # final distance is minimum over arms, expanded to [N, 1, R, R, H, W]
    distance = jnp.min(distance_branches,
                       axis=1, keepdims=True)*(patchres/(patchmax-patchmin))

    return distance

  def render_distance_tf(self, boundary_branches):
    return boundary_branches

  def render_boundaries(self,
                        vertex: jnp.ndarray,
                        boundaryangles: jnp.ndarray,
                        patchmin: float,
                        patchmax: float,
                        patchres: int,
                        delta: float = 0.005):
    """Render an image of the wedge boundaries over a square patch.

    Args:
      vertex: Array of shape [N, 2, H, W] containing the u and v coordinates
      of the vertices
      boundaryangles: Array of shape [N, 3, H, W] containing the three boundary
      angles (boundary-ray directions)
      patchmin: Minimum value of the patch
      patchmax: Maximum value of the patch
      patchres: Size of the patch
      delta: Delta value of the patch

    Returns:
      Array of shape [N, 1, R, R, H, W]
    """
    # coordinate grid of pixel locations
    yt = jnp.expand_dims(jnp.expand_dims(jnp.linspace(patchmin,
                                                      patchmax,
                                                      patchres), -1),
                         (0, 1, 4, 5))
    xt = jnp.expand_dims(jnp.expand_dims(jnp.linspace(patchmin,
                                                      patchmax,
                                                      patchres), 0),
                         (0, 1, 4, 5))

    x0 = jnp.expand_dims(vertex[:, 0, ...], (1, 2, 3))
    y0 = jnp.expand_dims(vertex[:, 1, ...], (1, 2, 3))

    cos_ba = jnp.cos(boundaryangles)
    sin_ba = jnp.sin(boundaryangles)

    # Use [1 / (1 + (x/opts.delta)**2 )] for the relaxed dirac distribution
    x = ((xt - x0)*cos_ba + (yt - y0)*sin_ba -
         jnp.sqrt((xt - x0)**2 + (yt - y0)**2))
    r = ((xt - x0)**2 + (yt - y0)**2)**(.5)

    patches = 1.0 / (1.0 + ((x * r) / delta)**2)

    standard_boundaries = jnp.max(patches, axis=1, keepdims=True)

    return standard_boundaries
