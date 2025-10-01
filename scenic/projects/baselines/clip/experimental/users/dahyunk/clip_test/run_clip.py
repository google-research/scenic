# Copyright 2025 The Scenic Authors.
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

"""Main file for CLIP tutorial."""
from clu import metric_writers
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.baselines.clip import model as clip
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  del rng, config, workdir, writer
  model_name = 'resnet_50'

  model = clip.MODELS[model_name]()
  clip_vars = clip.load_model_vars(model_name)
  model_bound = model.bind(clip_vars)

  tokenizer = clip_tokenizer.build_tokenizer()
  text = tokenizer('This is a cat.')
  # Note that different pretrained models run natively on different images
  # resolution (See `IMAGE_RESOLUTION` in model.py).
  image = jnp.zeros((1, 224, 224, 3))
  image = clip.normalize_image(image)

  # Or individually:
  encoded_text = model_bound.encode_text(text)
  encoded_image = model_bound.encode_image(image)
  print('image shape:', encoded_image.shape)
  print('text  shape:', encoded_text.shape)

if __name__ == '__main__':
  app.run(main=main)

