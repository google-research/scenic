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

r"""Main script for training Dense Video Captioning models."""

import os
from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.vid2seq import models
from scenic.projects.vid2seq import trainer
from scenic.projects.vid2seq.datasets.dense_video_captioning_tfrecord_dataset import get_datasets
# replace with the path to your JAVA bin location
JRE_BIN_JAVA = path_to_jre_bin_java

flags.DEFINE_string('jre_path', '',
                    'Path to JRE.')

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Callable[..., Any]:
  """Returns model class given its name."""
  if model_name == 'vid2seq':
    return models.DenseVideoCaptioningModel
  raise ValueError(f'Unrecognized model: {model_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the Vid2Seq project."""
  jave_jre = JRE_BIN_JAVA
  os.environ['JRE_BIN_JAVA'] = java_jre

  # ensure arguments match
  config.model.decoder.num_bins = config.dataset_configs.num_bins
  config.model.decoder.tmp_only = config.dataset_configs.tmp_only
  config.model.decoder.order = config.dataset_configs.order

  model_cls = get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset_dict = get_datasets(
      config,
      data_rng=data_rng)

  if config.num_training_epochs:
    trainer.train_and_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset_dict=dataset_dict,
        workdir=workdir,
        writer=writer)
  else:
    trainer.eval_only(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset_dict=dataset_dict,
        workdir=workdir,
        writer=writer)


if __name__ == '__main__':
  app.run(main=main)
