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

"""Main file for start training."""

from absl import flags
from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.knowledge_visual_language import trainer
from scenic.projects.knowledge_visual_language import trainer_memory
from scenic.projects.knowledge_visual_language import trainer_utils
from scenic.projects.knowledge_visual_language.data import cc12m_table_dataset  # pylint: disable=unused-import
from scenic.projects.knowledge_visual_language.data import vqa_dataset  # pylint: disable=unused-import
from scenic.projects.knowledge_visual_language.data import vqa_table_dataset  # pylint: disable=unused-import
from scenic.projects.knowledge_visual_language.data import web_image_text_generation_dataset  # pylint: disable=unused-import
from scenic.projects.knowledge_visual_language.data import wiki_image_text_generation_dataset  # pylint: disable=unused-import
from scenic.projects.knowledge_visual_language.data import wit_table_dataset  # pylint: disable=unused-import
from scenic.projects.knowledge_visual_language.models import fusion_in_decoder_soft
from scenic.projects.knowledge_visual_language.models import knowledge_fid
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for the knowledge project."""
  data_rng, rng = jax.random.split(rng)

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address
  )

  logging.info(FLAGS.dataset_service_address)
  logging.info('workdir= %s', workdir)
  logging.info('global_num_shards= %d', jax.device_count())
  logging.info('num_shards= %d', jax.local_device_count())
  logging.info('cpu info')
  logging.info(jax.local_devices(backend='cpu'))
  logging.info(jax.local_devices(backend='cpu'))
  logging.info('****************** Dataset metadata *****************')
  logging.info(dataset.meta_data)

  if config.update_num:
    trainer_utils.update_config(config, dataset.meta_data)
  if config.model_name == 'retrieval_image_captioner_soft':
    model_cls = fusion_in_decoder_soft.FIDSoftModel
    trainer.train_and_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer,
    )
  elif config.model_name == 'knowledge_fid':
    model_cls = knowledge_fid.KnowledgeFIDModel
    kb_datasets = {}
    for kb_dataset_name, kb_dataset_config in zip(
        config.kb_dataset_names, config.kb_dataset_configs
    ):
      kb_datasets[kb_dataset_name] = train_utils.get_dataset(
          config,
          data_rng,
          dataset_service_address=FLAGS.dataset_service_address,
          dataset_name=kb_dataset_name,
          dataset_configs=kb_dataset_config,
      )

    trainer_memory.train_and_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer,
        kb_datasets=kb_datasets,
    )
  else:
    raise ValueError(('Unknown model name %s' % config.model_name))

  return None


if __name__ == '__main__':
  app.run(main=main)
