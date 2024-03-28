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

"""Field of Junctions optimization reimplemented in JAX."""

import functools
import types

import jax
import jax.numpy as jnp
import optax


class FieldOfJunctions:
  """Field of Junctions optimization."""

  def __init__(self, img, opts):
    """Inputs.

    Args:
      img: Input image: a numpy array of shape [height, width, channels]
      opts: Object with the following attributes:
            patchsize                  Patch size
            stride                     Stride for junctions (e.g. opts.stride ==
                                        1 is a dense field of junctions)
            eta                        Width parameter for Heaviside functions
            delta                      Width parameter for boundary maps
            lr_angles                  Angle learning rate
            lr_x0y0                    Vertex position learning rate
            lambda_boundary_final      Final value of spatial boundary
                                        consistency term
            lambda_color_final         Final value of spatial color consistency
                                        term
            nvals                      Number of values to query in Algorithm 2
                                        from the paper
            num_initialization_iters   Number of initialization iterations
            num_refinement_iters       Number of refinement iterations
            greedy_step_every_iters    Frequency of "greedy" iteration (applying
                                        Algorithm 2 with consistency)
            parallel_mode              Whether or not to run Algorithm 2 in
                                        parallel over all `nvals` values.
    """

    # Save opts
    self.opts = opts

    # Get image dimensions
    self.height, self.width, self.channels = img.shape

    # Number of patches (throughout the documentation hpatches and wpatches
    # are denoted by H' and W' resp.)
    self.hpatches = (self.height - opts.patchsize) // opts.stride + 1
    self.wpatches = (self.width - opts.patchsize) // opts.stride + 1

    # Set initial lmbda_boundary and lmbda_color variables
    self.lmbda_boundary = 0
    self.lmbda_color = 0

    # Store total number of iterations (initialization + refinement)
    self.num_iters = opts.num_initialization_iters + opts.num_refinement_iters

    # Split image into overlapping patches, creating a array of shape
    # N, channels, R, R, hpatches, wpatches
    t_img = jnp.expand_dims(
        jnp.array(img).transpose(2, 0, 1), 0
    )  # input image, shape [1, channels, height, width]
    self.img_patches = self.unfold(t_img)

    # Create  variables for angles and vertex position for each patch
    self.angles = jnp.zeros(
        (1, 3, self.hpatches, self.wpatches), dtype=jnp.float32
    )
    self.x0y0 = jnp.zeros(
        (1, 2, self.hpatches, self.wpatches), dtype=jnp.float32
    )

    val = types.SimpleNamespace(
        shape=(1, 1, self.height, self.width), dtype=jnp.dtype(jnp.float32)
    )
    self.num_patches = self.fold(
        jnp.ones(
            (1, 1, opts.patchsize, opts.patchsize, self.hpatches,
             self.wpatches)), val).reshape(self.height, self.width)

    # Create local grid within each patch
    y, x = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, opts.patchsize),
        jnp.linspace(-1.0, 1.0, opts.patchsize)
    )
    self.x = x.reshape(1, opts.patchsize, opts.patchsize, 1, 1)
    self.y = y.reshape(1, opts.patchsize, opts.patchsize, 1, 1)

    # Optimization parameters
    adam_beta1 = 0.5
    adam_beta2 = 0.99
    adam_eps = 1e-08

    optimizer_angles = optax.adam(
        opts.lr_angles, adam_beta1, adam_beta2, eps=adam_eps
    )
    opt_state_angles = optimizer_angles.init(self.angles)

    optimizer_x0y0 = optax.adam(
        opts.lr_x0y0, adam_beta1, adam_beta2, eps=adam_eps
    )
    opt_state_x0y0 = optimizer_x0y0.init(self.x0y0)

    self.optimizers = [optimizer_angles, optimizer_x0y0]
    self.opt_states = [opt_state_angles, opt_state_x0y0]

    # # Values to search over in Algorithm 2: [0, 2pi) for angles, [-3, 3] for
    # vertex position.
    self.angle_range = jnp.linspace(0.0, 2 * jnp.pi, opts.nvals + 1)[
        : opts.nvals
    ]
    self.x0y0_range = jnp.linspace(-3.0, 3.0, opts.nvals)

    # Save current global image and boundary map (initially None)
    self.global_image = jnp.zeros(
        (1, 3, self.height, self.width), dtype=jnp.float32
    )
    self.global_boundaries = jnp.zeros(
        (1, 1, self.height, self.width), dtype=jnp.float32
    )

  def step(
      self, iteration, angles, x0y0, global_image, global_boundaries, opt_states
  ):
    """Performs one step of initialization or refinement.

    Args:
      iteration: Iteration number (integer)
      angles: Array of shape [N, 3, H', W'] with the angles for each patch
      x0y0: Array of shape [N, 2, H', W'] with the vertex positions for each
      patch
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      opt_states: Optimization states for each patch

    Returns:
      angles: Updated angles for each patch
      x0y0: Updated vertex positions for each patch
      global_image: Updated global image
      global_boundaries: Updated global boundaries
      opt_states: Updated optimization states for each patch
    """

    # Linearly increase lambda from 0 to lambda_boundary_final and
    # lambda_color_final
    if self.opts.num_refinement_iters <= 1:
      factor = 0.0
    else:
      factor = max([
          0,
          (iteration - self.opts.num_initialization_iters)
          / (self.opts.num_refinement_iters - 1),
      ])

    lmbda_boundary = factor * self.opts.lambda_boundary_final
    lmbda_color = factor * self.opts.lambda_color_final

    if (
        iteration < self.opts.num_initialization_iters
        or (iteration - self.opts.num_initialization_iters + 1)
        % self.opts.greedy_step_every_iters
        == 0
    ):
      angles, x0y0, global_image, global_boundaries = self.initialization_step(
          angles,
          x0y0,
          global_image,
          global_boundaries,
          lmbda_boundary,
          lmbda_color,
      )
    else:
      angles, x0y0, global_image, global_boundaries, opt_states = (
          self.refinement_step(
              angles,
              x0y0,
              global_image,
              global_boundaries,
              lmbda_boundary,
              lmbda_color,
              opt_states,
          )
      )

    return angles, x0y0, global_image, global_boundaries, opt_states

  @functools.partial(jax.jit, static_argnums=(0,))
  def initialization_step(
      self,
      angles,
      x0y0,
      global_image,
      global_boundaries,
      lmbda_boundary,
      lmbda_color,
  ):
    """Perform a single coordinate descent step.

    Implements a heuristic for searching along the three junction angles after
    updating each of the five parameters. The original value is included in the
    search, so the extra step is guaranteed to obtain a better (or equally-good)
    set of parameters.

    Args:
      angles: Array of shape [N, 3, H', W'] with the angles for each patch
      x0y0: Array of shape [N, 2, H', W'] with the vertex positions for each
        patch
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      lmbda_boundary: Spatial consistency boundary loss weight
      lmbda_color: Spatial consistency color loss weight

    Returns:
      angles: Updated angles for each patch
      x0y0: Updated vertex positions for each patch
      global_image: Updated global image
      global_boundaries: Updated global boundaries
    """

    params = jnp.concatenate([angles, x0y0], axis=1)

    # Run one step of Algorithm 2, sequentially improving each coordinate
    for i in range(5):
      # Repeat the set of parameters `nvals` times along 0th dimension
      params_query = jnp.repeat(params, self.opts.nvals, axis=0)
      param_range = self.angle_range if i < 3 else self.x0y0_range
      params_query = params_query.at[:, i, :, :].set(
          params_query[:, i, :, :] + param_range.reshape(-1, 1, 1)
      )

      best_ind = self.get_best_inds(
          params_query,
          global_image,
          global_boundaries,
          lmbda_boundary,
          lmbda_color,
      )

      # Update parameters
      params = params.at[0, i, :, :].set(
          params_query[
              best_ind.reshape(self.hpatches, self.wpatches),
              i,
              jnp.arange(self.hpatches).reshape(-1, 1),
              jnp.arange(self.wpatches).reshape(1, -1),
          ]
      )

    # Heuristic for accelerating convergence(not necessary but sometimes
    # helps):
    # Update x0 and y0 along the three optimal angles (search over a line
    # passing through current x0, y0)
    for i in range(3):
      params_query = jnp.repeat(params, self.opts.nvals, axis=0)
      params_query = params_query.at[:, 3, :, :].set(
          params[:, 3, :, :]
          + jnp.cos(params[:, i, :, :])
          * jnp.expand_dims(self.x0y0_range, [1, 2]).reshape(-1, 1, 1)
      )
      params_query = params_query.at[:, 4, :, :].set(
          params[:, 4, :, :]
          + jnp.sin(params[:, i, :, :])
          * jnp.expand_dims(self.x0y0_range, [1, 2]).reshape(-1, 1, 1)
      )

      best_ind = self.get_best_inds(
          params_query,
          global_image,
          global_boundaries,
          lmbda_boundary,
          lmbda_color,
      )

      # Update vertex positions of parameters
      for j in range(3, 5):
        params = params.at[:, j, :, :].set(
            params_query[
                jnp.expand_dims(best_ind, 0).reshape(
                    1, self.hpatches, self.wpatches
                ),
                j,
                jnp.expand_dims(jnp.arange(self.hpatches), [0, 2]).reshape(
                    1, -1, 1
                ),
                jnp.expand_dims(jnp.arange(self.wpatches), [0, 1]).reshape(
                    1, 1, -1
                ),
            ]
        )

    # Update angles and vertex position using the best values found
    angles = params[:, :3, :, :]
    x0y0 = params[:, 3:, :, :]

    # Update global boundaries and image
    dists, _, patches = self.get_dists_and_patches(
        params, global_image, lmbda_color
    )
    global_image = self.local2global(patches)
    global_boundaries = self.local2global(self.dists2boundaries(dists))

    return angles, x0y0, global_image, global_boundaries

  def get_best_inds(
      self, params, global_image, global_boundaries, lmbda_boundary, lmbda_color
      ):
    """Compute the best index for each patch.

    Has two possible modes determined by self.opts.parallel_mode: 1) When True,
    all N values are computed in parallel (generally faster, requires more
    memory) 2) When False, the values are computed sequentially (generally
    slower, requires less memory)

    Args:
      params: Array of shape [N, 5, H', W'] holding N field of junctions
        parameters. Each 5-vector has format (angle1, angle2, angle3, x0, y0).
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      lmbda_boundary: Spatial consistency boundary loss weight
      lmbda_color: Spatial consistency color loss weight

    Returns:
      Array of shape [H', W'] with each value in {0, ..., N-1} holding the
      index of the best junction parameters at that patch position.
    """

    if self.opts.parallel_mode:
      dists, colors, smoothpatches = self.get_dists_and_patches(
          params, global_image, lmbda_color
      )
      loss_per_patch = self.get_loss(
          dists,
          colors,
          smoothpatches,
          global_image,
          global_boundaries,
          lmbda_boundary,
          lmbda_color,
      )
      best_ind = jnp.argmin(loss_per_patch, axis=0)

    else:
      # First initialize arrays
      best_ind = jnp.zeros((self.hpatches, self.wpatches), dtype=jnp.int64)
      best_loss_per_patch = jnp.zeros((self.hpatches, self.wpatches)) + 1e10

      # Now fill arrays by iterating over the junction dimension and choosing
      # the best junction parameters
      for n in range(params.shape[0]):
        dists, colors, smoothpatches = self.get_dists_and_patches(
            params[n : n + 1, :, :, :], global_image, lmbda_color
        )
        loss_per_patch = self.get_loss(
            dists,
            colors,
            smoothpatches,
            global_image,
            global_boundaries,
            lmbda_boundary,
            lmbda_color,
        )

        improved_inds = loss_per_patch[0] < best_loss_per_patch
        best_ind = jnp.where(
            improved_inds, jnp.array(n, dtype=jnp.int64), best_ind
        )
        best_loss_per_patch = jnp.where(
            improved_inds, loss_per_patch, best_loss_per_patch
        )

    return best_ind

  def unfold(self, im):
    """Extract patches from an image.

    Args:
      im: Array of shape [N, C, H, W]

    Returns:
      Array of shape [N, C, R, R, H', W'] containing all image patches.
      E.g. [k,l,:,:,i,j] is the lth channel of the (i,j)th patch of the kth
      image
    """

    return jax.lax.conv_general_dilated_patches(
        im,
        filter_shape=[self.opts.patchsize, self.opts.patchsize],
        window_strides=[self.opts.stride, self.opts.stride],
        padding='VALID',
    ).reshape([
        -1,
        im.shape[1],
        self.opts.patchsize,
        self.opts.patchsize,
        self.hpatches,
        self.wpatches,
    ])

  def fold(self, patches, val):
    """Fold patches into a single image.

    Args:
      patches: Array of shape [N, C, R, R, H', W']
      val: Container of shape [N, C, H, W]

    Returns:
      Array of shape [N, C, H, W] containing the folded image patches.
    """

    f_transpose = jax.linear_transpose(self.unfold, val)

    return f_transpose(patches)[0]

  def get_dists_and_patches(self, params, global_image, lmbda_color):
    """Compute distance functions and piecewise-constant patches.

    Args:
      params: Array of shape [N, 5, H', W'] holding N field of junctions
        parameters. Each 5-vector has format (angle1, angle2, angle3, x0, y0)
      global_image: Array of shape [N, C, H, W] with the global image
      lmbda_color: Factor between 0 and 1 that dictates how to mix global image
      with the input noisy image when calculating wedge colors

    Returns:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
        distance functions for every patch
      colors: Array of shape [N, C, 3, H', W'] storing the colors at each wedge
      patches: Array of shape [N, C, H', W'] with the constant color function
      at each of the 3 wedges
    """

    # Get dists
    dists = self.params2dists(params)  # shape [N, 2, R, R, H', W']

    # Get wedge indicator functions
    wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

    curr_global_image_patches = self.unfold(global_image)

    numerator = (
        jnp.expand_dims(
            self.img_patches + lmbda_color * curr_global_image_patches, 2
        )
        * jnp.expand_dims(wedges, 1)
    ).sum([3, 4])
    denominator = (1.0 + lmbda_color) * jnp.expand_dims(wedges.sum([2, 3]), 1)
    colors = numerator / (denominator + 1e-10)

    # Fill wedges with optimal colors
    patches = (
        jnp.expand_dims(wedges, 1) * jnp.expand_dims(colors, [3, 4])
    ).sum(axis=2)

    return dists, colors, patches

  def params2dists(self, params, tau=1e-1):
    """Compute distance functions d_{13}, d_{12}.

    Args:
      params: Array of shape [N, 5, H', W'] holding N field of junctions
        parameters. Each 5-vector has format (angle1, angle2, angle3, x0, y0)
      tau: Small constant to add to the gradient of the distance functions

    Returns:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
        distance functions for every patch
    """

    x0 = jnp.expand_dims(params[:, 3, :, :], [1, 2])  # shape [N, 1, 1, H', W']
    y0 = jnp.expand_dims(params[:, 4, :, :], [1, 2])  # shape [N, 1, 1, H', W']

    # Sort so angle1 <= angle2 <= angle3 (mod 2pi)
    angles = jnp.remainder(params[:, :3, :, :], 2 * jnp.pi)

    angles = jnp.sort(angles, axis=1)

    angle1 = jnp.expand_dims(
        angles[:, 0, :, :], [1, 2]
    )  # shape [N, 1, 1, H', W']
    angle2 = jnp.expand_dims(
        angles[:, 1, :, :], [1, 2]
    )  # shape [N, 1, 1, H', W']
    angle3 = jnp.expand_dims(
        angles[:, 2, :, :], [1, 2]
    )  # shape [N, 1, 1, H', W']

    # Define another angle halfway between angle3 and angle1, clockwise from
    # angle3. This isn't critical but it seems a bit more stable for computing
    # gradients.
    angle4 = 0.5 * (angle1 + angle3) + jnp.where(
        jnp.remainder(0.5 * (angle1 - angle3), 2 * jnp.pi) >= jnp.pi,
        jnp.ones_like(angle1) * jnp.pi,
        jnp.zeros_like(angle1),
    )

    def g(dtheta):
      # Map from [0, 2pi] to [-1, 1]
      return (dtheta / jnp.pi - 1.0) ** 35

    # Compute the two distance functions
    sgn42 = jnp.where(
        jnp.remainder(angle2 - angle4, 2 * jnp.pi) < jnp.pi,
        jnp.ones_like(angle2),
        -jnp.ones_like(angle2),
    )
    tau42 = g(jnp.remainder(angle2 - angle4, 2 * jnp.pi)) * tau

    dist42 = (
        sgn42
        * jnp.minimum(
            sgn42
            * (
                -jnp.sin(angle4) * (self.x - x0)
                + jnp.cos(angle4) * (self.y - y0)
            ),
            -sgn42
            * (
                -jnp.sin(angle2) * (self.x - x0)
                + jnp.cos(angle2) * (self.y - y0)
            ),
        )
        + tau42
    )

    sgn13 = jnp.where(
        jnp.remainder(angle3 - angle1, 2 * jnp.pi) < jnp.pi,
        jnp.ones_like(angle3),
        -jnp.ones_like(angle3),
    )
    tau13 = g(jnp.remainder(angle3 - angle1, 2 * jnp.pi)) * tau
    dist13 = (
        sgn13
        * jnp.minimum(
            sgn13
            * (
                -jnp.sin(angle1) * (self.x - x0)
                + jnp.cos(angle1) * (self.y - y0)
            ),
            -sgn13
            * (
                -jnp.sin(angle3) * (self.x - x0)
                + jnp.cos(angle3) * (self.y - y0)
            ),
        )
        + tau13
    )

    return jnp.stack([dist13, dist42], axis=1)

  def dists2indicators(self, dists):
    """Computes the indicator functions from the distance functions.

    Args:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
      distance functions for every patch

    Returns:
      Array of shape [N, 3, R, R, H', W'] with samples of the three
      indicator functions for every patch
    """
    # Apply smooth Heaviside function to distance functions
    hdists = 0.5 * (1.0 + (2.0 / jnp.pi) * jnp.arctan(dists / self.opts.eta))

    # Convert Heaviside functions into wedge indicator functions
    return jnp.stack(
        [
            1.0 - hdists[:, 0, :, :, :, :],
            hdists[:, 0, :, :, :, :] * (1.0 - hdists[:, 1, :, :, :, :]),
            hdists[:, 0, :, :, :, :] * hdists[:, 1, :, :, :, :],
        ],
        axis=1,
    )

  def get_loss(
      self,
      dists,
      colors,
      patches,
      global_image,
      global_boundaries,
      lmbda_boundary,
      lmbda_color,
  ):
    """Compute the objective of the model (see Equation 8 of the paper).

    Args:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
      distance functions for every patch
      colors: Array of shape [N, C, 3, H', W'] storing the colors at each wedge
      patches: Array of shape [N, C, H', W'] with the constant color function
      at each of the 3 wedges
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      lmbda_boundary: Spatial consistency boundary loss weight
      lmbda_color: Spatial consistency color loss weight

    Returns:
      Array of shape [N, H', W'] with the loss at each patch
    """
    # Compute negative log-likelihood for each patch (shape [N, H', W'])
    loss_per_patch = (
        ((self.img_patches - patches) ** 2).mean(-3).mean(-3).sum(1)
    )

    # Add spatial consistency loss for each patch, if lambda > 0
    loss_per_patch = (
        loss_per_patch
        + lmbda_boundary
        * self.get_boundary_consistency_term(dists, global_boundaries)
    )
    loss_per_patch = (
        loss_per_patch
        + lmbda_color
        * self.get_color_consistency_term(dists, colors, global_image)
    )

    return loss_per_patch

  def local2global(self, patches):
    """Compute average value for each pixel over all patches containing it.

    For example, this can be used to compute the global boundary maps, or the
    boundary-aware smoothed image.

    Args:
      patches: Array of shape [N, C, H', W'] with the constant color function
      at each of the 3 wedges
      patches[n, :, :, :, i, j] is an RxR C-channel patch at the (i, j)th
      spatial position of the nth entry.

    Returns:
      Array of shape [N, C, H, W] of averages over all patches containing
      each pixel.
    """
    batch = patches.shape[0]
    channels = patches.shape[1]

    val = types.SimpleNamespace(
        shape=(batch, channels, self.height, self.width),
        dtype=jnp.dtype(jnp.float32),
    )

    return jnp.divide(
        self.fold(
            patches,
            val).reshape(batch,
                         channels,
                         self.height,
                         self.width), jnp.expand_dims(self.num_patches, [0, 1]))

  def dists2boundaries(self, dists):
    """Compute boundary map for each patch, given distance functions.

    The width of the boundary is determined by self.opts.delta.

    Args:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
      distance functions for every patch

    Returns:
      Array of shape [N, 1, R, R, H', W'] with values of boundary map for
      every patch
    """
    # Find places where either distance transform is small, except where d1 > 0
    # and d2 < 0
    d1 = dists[:, 0:1, :, :, :, :]
    d2 = dists[:, 1:2, :, :, :, :]
    minabsdist = jnp.where(
        d1 < 0.0,
        -d1,
        jnp.where(d2 < 0.0, jnp.minimum(d1, -d2), jnp.minimum(d1, d2)),
    )

    return 1.0 / (1.0 + (minabsdist / self.opts.delta) ** 2)

  @functools.partial(jax.jit, static_argnums=(0,))
  def refinement_step(
      self,
      angles,
      x0y0,
      global_image,
      global_boundaries,
      lmbda_boundary,
      lmbda_color,
      opt_states,
  ):
    """Perform a single refinement step.

    Args:
      angles: Array of shape [N, 3, H', W'] with the angles for each patch
      x0y0: Array of shape [N, 2, H', W'] with the vertex positions for each
      patch
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      lmbda_boundary: Spatial consistency boundary loss weight
      lmbda_color: Spatial consistency color loss weight
      opt_states: Optimization states for each patch

    Returns:
      angles: Array of shape [N, 3, H', W'] with the updated angles for each
      patch
      x0y0: Array of shape [N, 2, H', W'] with the updated vertex positions for
      each patch
      global_image: Array of shape [N, C, H, W] with updated global image
      global_boundaries: Array of shape [N, 1, H, W] with updated global
      boundaries
      opt_states: Updated optimization states for each patch
    """
    dists, _, patches, angles, x0y0, opt_states = self.refinement_loss(
        angles,
        x0y0,
        global_image,
        global_boundaries,
        lmbda_boundary,
        lmbda_color,
        opt_states,
    )

    global_image = self.local2global(patches)
    global_boundaries = self.local2global(self.dists2boundaries(dists))

    return angles, x0y0, global_image, global_boundaries, opt_states

  def avg_loss(
      self,
      angles,
      x0y0,
      global_image,
      global_boundaries,
      lmbda_boundary,
      lmbda_color,
  ):
    """Calculates the average loss.

    Args:
      angles: Array of shape [N, 3, H', W'] with angles for each patch
      x0y0: Array of shape [N, 2, H', W'] with vertex positions for each patch
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      lmbda_boundary: Spatial consistency boundary loss weight
      lmbda_color: Spatial consistency color loss weight

    Returns:
      Average loss
    """
    params = jnp.concatenate([angles, x0y0], axis=1)

    # Compute distance functions, colors, and junction patches
    dists, colors, patches = self.get_dists_and_patches(
        params, global_image, lmbda_color
    )

    # Compute average loss
    return self.get_loss(
        dists,
        colors,
        patches,
        global_image,
        global_boundaries,
        lmbda_boundary,
        lmbda_color,
    ).mean()

  def refinement_loss(
      self,
      angles,
      x0y0,
      global_image,
      global_boundaries,
      lmbda_boundary,
      lmbda_color,
      opt_states,
  ):
    """Calculates the refinement loss.

    Args:
      angles: Array of shape [N, 3, H', W'] with the angles for each patch
      x0y0: Array of shape [N, 2, H', W'] with the vertex positions for each
      patch
      global_image: Array of shape [N, C, H, W] with the global image
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries
      lmbda_boundary: Spatial consistency boundary loss weight
      lmbda_color: Spatial consistency color loss weight
      opt_states: The optimizer’s states

    Returns:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
      distance functions for every patch
      colors: Array of shape [N, C, 3, H', W'] storing the colors at each wedge
      patches: Array of shape [N, C, H', W'] with the constant color function
      at each of the 3 wedges
      angles: Array of shape [N, 3, H', W'] with the angles for each patch
      x0y0: Array of shape [N, 2, H', W'] with the vertex positions for each
      patch
      opt_states: The optimizer’s states
    """

    grad_angles, grad_x0y0 = jax.grad(self.avg_loss, argnums=[0, 1])(
        angles,
        x0y0,
        global_image,
        global_boundaries,
        lmbda_boundary,
        lmbda_color,
    )

    updates_angles, opt_states[0] = self.optimizers[0].update(
        grad_angles, opt_states[0]
    )
    angles = optax.apply_updates(angles, updates_angles)

    updates_x0y0, opt_states[1] = self.optimizers[1].update(
        grad_x0y0, opt_states[1]
    )
    x0y0 = optax.apply_updates(x0y0, updates_x0y0)

    # Update global boundaries and image
    params = jnp.concatenate([angles, x0y0], axis=1)
    dists, colors, patches = self.get_dists_and_patches(
        params, global_image, lmbda_color
    )

    return dists, colors, patches, angles, x0y0, opt_states

  def get_boundary_consistency_term(self, dists, global_boundaries):
    """Compute the boundary consistency loss.

    Args:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
      distance functions for every patch
      global_boundaries: Array of shape [N, 1, H, W] with the global boundaries

    Returns:
      consistency: Array of shape [N, H', W'] with the boundary consistency loss
      at each patch
    """

    # Split global boundaries into patches
    curr_global_boundaries_patches = self.unfold(global_boundaries)

    # Get local boundaries defined using the queried parameters (defined by
    # `dists`)
    local_boundaries = self.dists2boundaries(dists)

    # Compute consistency term
    consistency = (
        ((local_boundaries - curr_global_boundaries_patches) ** 2)
        .mean(2).mean(2)
    )

    return consistency[:, 0, :, :]

  def get_color_consistency_term(self, dists, colors, global_image):
    """Compute the spatial color consistency loss.

    Args:
      dists: Array of shape [N, 2, R, R, H', W'] with samples of the two
      distance functions for every patch
      colors: Array of shape [N, C, 3, H', W'] storing the colors at each wedge
      global_image: Array of shape [N, C, H, W] with the global image

    Returns:
      Array of shape [N, H', W'] with the color consistency loss at each
      patch
    """

    # Split global image into patches
    curr_global_image_patches = self.unfold(global_image)

    wedges = self.dists2indicators(dists)

    # Compute the color consistency loss
    consistency = (
        (jnp.expand_dims(wedges, 1)
         * (jnp.expand_dims(colors, [3, 4])
            - jnp.expand_dims(curr_global_image_patches, 2)
            ) ** 2).mean(2).mean(2).sum(1).sum(1))

    return consistency
