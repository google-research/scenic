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

"""Evaluation script for the PixelLLM.

This file is modified from scenic CenterNet code at
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/
centernet/evaluate.py
"""

import functools
import time
from typing import Any, Optional

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.core import frozen_dict
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils


def inference_on_all_datasets(
    model: Any,
    train_state: train_utils.TrainState,
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
    eval_batch_size: int = 1,
    is_host: bool = False,
    save_dir: str = '',
    step: Optional[int] = None,
    config: ml_collections.ConfigDict = ml_collections.ConfigDict(),
    ) -> Any:
  """The main evaluation loop. Run evaluation on the whole validation set.

  Args:
    model: Scenic basemodel (an instance of nn.Module).
    train_state: train_state that contains the model parameters.
    dataset: The dataset that has valid_iter and meta_data.
    writer: metric_writers.MetricWriter
    eval_batch_size: integer. Batch size per-device in evaluation.
    is_host: bool: whether its the host machine. During multi-machine training,
      we only hold the evaluating data in one of the machines. The machine with
      `jax.process_index() == 0` sets `is_host` to True and will gather data
      from other machines and do the evaluation. Other machines set `is_host` as
      False.
    save_dir: string: where to save the json prediction
    step: Optional integer of the training step. The step is appended to the
      serialised results if provided.
    config: config dict

  Returns:
    evaluation results.
  """
  eval_summary = None
  assert isinstance(
      dataset.valid_iter, dict
  ), 'Only dict valid_iter are supported.'
  for ds_name, ds_iter in dataset.valid_iter.items():
    if ds_name in [
        'refcoco_unc_validation',
        'refcoco_unc_testA',
        'refcoco_unc_testB',
        'refcocog_umd_validation',
        'refcocog_umd_test',
        'refcocoplus_unc_validation',
        'refcocoplus_unc_testA',
        'refcocoplus_unc_testB',
    ]:
      eval_func = inference_on_dataset_refer
    elif ds_name in ['vg_densecap_test']:
      eval_func = inference_on_dataset_densecap
    elif ds_name in ['vg_loca_test', 'refcocog_umd_loca_validation']:
      eval_func = inference_on_dataset_loca
    elif ds_name in ['ln_coco_trace_val']:
      eval_func = inference_on_dataset_trace
    elif ds_name in ['coco_captions_val']:
      eval_func = inference_on_dataset_caption
    else:
      raise ValueError('Unsupported dataset name: %s' % ds_name)
    meta_data = dataset.meta_data.copy()
    meta_data['num_eval_examples'] = dataset.meta_data['num_eval_examples'][
        ds_name
    ]
    eval_dataset = dataset_utils.Dataset(
        valid_iter=ds_iter, meta_data=meta_data
    )
    start_time = time.time()
    eval_results, eval_metrics = eval_func(
        model,
        train_state,
        eval_dataset,
        dataset_name=ds_name,
        eval_batch_size=eval_batch_size,
        is_host=is_host,
        save_dir=save_dir,
        step=step,
        config=config,
    )
    eval_summary = train_utils.log_eval_summary(
        step=step if step is not None else 0,
        eval_metrics=eval_metrics,
        extra_eval_summary=eval_results,
        prefix=f'{ds_name}_valid',
        writer=writer,
    )
    duration = time.time() - start_time
    logging.info('Done with %s evaluation: %.4f sec.', ds_name, duration)

  return eval_summary
