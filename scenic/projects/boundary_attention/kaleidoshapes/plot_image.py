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

""""Visualize kaleidoshapes data."""

import matplotlib.pyplot as plt


def plot_image(imagedict):
  """Visualize a single kaleidoshapes image."""

  plt.figure(figsize=(20, 10))
  plt.subplot(141)
  plt.imshow(imagedict['image'])
  plt.xticks([])
  plt.yticks([])
  plt.title(f"Image ({imagedict['num_shapes']})")
  plt.ylabel('Raw shapes')

  plt.subplot(142)
  plt.imshow(imagedict['boundaries'], cmap='binary')
  for intersection in imagedict['intersections']* 320:
    plt.plot(intersection[0], intersection[1], 'rx')
  for vertex in imagedict['vertices']* 320:
    plt.plot(vertex[0], vertex[1], 'bx')
  plt.xticks([])
  plt.yticks([])
  plt.title(f"Boundaries ({imagedict['num_shapes']})")

  plt.subplot(143)
  plt.imshow(imagedict['segments'], cmap='gray')
  plt.xticks([])
  plt.yticks([])
  plt.title(f"Segments ({imagedict['num_shapes']})")

  plt.subplot(144)
  plt.imshow(imagedict['distances'])
  plt.xticks([])
  plt.yticks([])
  plt.title(f"Distance ({imagedict['num_shapes']})")
  plt.colorbar(fraction=0.046, pad=0.04)
