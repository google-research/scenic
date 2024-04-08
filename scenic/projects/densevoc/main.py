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

"""Main file for DenseVOC."""
import os
from typing import Any

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.densevoc import evaluate
from scenic.projects.densevoc import input_pipeline
from scenic.projects.densevoc import trainer
from scenic.projects.densevoc.modeling import densevoc_model
from scenic.projects.densevoc.modeling import grit
# replace with the path to your JAVA bin location
JRE_BIN_JAVA = path_to_jre_bin_java

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Any:
  """Returns model class given its name."""
  if model_name == 'grit':
    return grit.GRiTModel
  elif model_name == 'densevoc':
    return densevoc_model.DenseVOCModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the CenterNet project."""
  jave_jre = JRE_BIN_JAVA
  os.environ['JRE_BIN_JAVA'] = java_jre
  model_cls = get_model_cls(config.model.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = input_pipeline.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  eval_only = config.get('eval_only', False)
  if eval_only:
    evaluate.evaluate(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)
  else:
    trainer.train_and_evaluate(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)


if __name__ == '__main__':
  app.run(main=main)
