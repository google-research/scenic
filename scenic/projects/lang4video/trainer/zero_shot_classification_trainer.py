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

"""Scenic trainer for zero-shot text-based visual classification evaluation.

Note this "trainer" doesn't actually train but just evaluates.
"""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Optional

from absl import logging
from clu import metric_writers
from flax import jax_utils
import flax.core
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import model_utils
from scenic.projects.lang4video import util
from scenic.projects.lang4video.model.image_text_model import ImageTextModel
from scenic.projects.lang4video.trainer.train_utils import axis_name_exists
from scenic.projects.lang4video.trainer.train_utils import compute_mask
from scenic.projects.lang4video.trainer.train_utils import get_cached_fn
from scenic.projects.lang4video.trainer.train_utils import get_epoch_steps
from scenic.projects.lang4video.trainer.train_utils import get_input_spec
from scenic.projects.lang4video.trainer.train_utils import init_encoder
from scenic.projects.lang4video.trainer.train_utils import InputSpec
from scenic.projects.lang4video.trainer.train_utils import is_video_input
from scenic.projects.lang4video.trainer.train_utils import log_eval_summary
from scenic.projects.lang4video.trainer.train_utils import NUM_DEVICES_AXIS_NAME
from scenic.projects.lang4video.trainer.train_utils import pad_array_to_be_divisible
from scenic.projects.lang4video.trainer.train_utils import partial_with_cache
from scenic.projects.lang4video.trainer.train_utils import split_in_batches
from scenic.train_lib import train_utils
from tqdm.auto import tqdm

# TODO(sacastro): support multiple clips.


# We pass the model and not directly the encoder because it's hashable.
def eval_step(
    train_state: train_utils.TrainState,
    visual: jnp.ndarray,  # Shape: (N, F, H, W, C) if `is_video`, w/o F if not.
    target: jnp.ndarray,  # Shape: (N) as int or (N, K) as float
    batch_mask: jnp.ndarray,  # Shape: (N,)
    *,
    model: ImageTextModel,
    encoded_classes: jnp.ndarray,  # Shape: (K, E)
    is_video: bool,
    debug: bool = False,
) -> Mapping[str, tuple[float, int]]:
  """Runs a single step of evaluation."""
  encoded_visual = model.flax_model.apply(
      variables={
          'params': train_state.params,
          **train_state.model_state,
      },
      method=model.flax_model.encode_video
      if is_video else model.flax_model.encode_image,
      **{'video' if is_video else 'image': visual},
      train=False,
      debug=debug)

  score = model.flax_model.compute_similarity(encoded_visual, encoded_classes)

  if target.ndim == score.ndim - 1:
    target = jax.nn.one_hot(target, len(encoded_classes))
  assert target.ndim == score.ndim

  metrics = {
      'accuracy@1':
          model_utils.weighted_top_one_correctly_classified(
              score, target, weights=batch_mask),
      'accuracy@5':
          model_utils.weighted_topk_correctly_classified(
              score, target, k=5, weights=batch_mask),
      'loss':
          optax.softmax_cross_entropy(score, target) * batch_mask,
  }

  actual_local_batch_size = batch_mask.sum()
  for k, v in metrics.items():
    metrics[k] = (v, actual_local_batch_size)

  if axis_name_exists(NUM_DEVICES_AXIS_NAME):
    actual_batch_size = jax.lax.psum(
        actual_local_batch_size, axis_name=NUM_DEVICES_AXIS_NAME)
    for k, v in metrics.items():
      metrics[k] = (jax.lax.psum(v[0].sum(), axis_name=NUM_DEVICES_AXIS_NAME),
                    actual_batch_size)

  return metrics  # pytype: disable=bad-return-type  # jax-ndarray


def compute_class_text_embeddings(
    train_state: train_utils.TrainState,
    class_texts: jnp.ndarray,  # Shape: (C, T, L)
    mask: Optional[jnp.ndarray] = None,  # Shape: (C, T, L)
    *,
    model: ImageTextModel,
    batch_size: int,
    debug: bool = False,
) -> jnp.ndarray:  # Shape: (C, E)
  """Computes the class text embeddings."""
  num_classes = class_texts.shape[0]
  class_texts = class_texts.reshape(-1, class_texts.shape[-1])
  if mask is not None:
    mask = mask.reshape(-1, mask.shape[-1])

  def _compute_class_text_embeddings_batch(
      args,  # `(batch, mask)` with shape: (N, L)
  ) -> jnp.ndarray:  # Shape: (N, E)
    batch, mask_ = args
    return model.flax_model.apply(
        variables={
            'params': train_state.params,
            **train_state.model_state,
        },
        method=model.flax_model.encode_text,
        text=batch,
        mask=mask_,
        train=False,
        debug=debug)

  # We batch to not run OOM.
  encoded_text = split_in_batches(
      _compute_class_text_embeddings_batch,
      batch_size=batch_size)((class_texts, mask))

  encoded_text = encoded_text.reshape(num_classes, -1,
                                      encoded_text.shape[-1]).mean(axis=1)

  if axis_name_exists(NUM_DEVICES_AXIS_NAME):
    encoded_text = jax.lax.all_gather(encoded_text, NUM_DEVICES_AXIS_NAME)
    encoded_text = encoded_text.reshape(-1, encoded_text.shape[-1])

  return encoded_text


@get_cached_fn
def _create_model_and_train_state(
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    model_cls: type[ImageTextModel],
    input_spec: InputSpec,
    is_video: bool,
    rng: Optional[jnp.ndarray] = None,
) -> tuple[ImageTextModel, train_utils.TrainState]:
  """Creates the model and train state."""
  model = model_cls(config, dataset.meta_data)
  encoder = model.flax_model

  params, model_state = init_encoder(
      encoder=encoder,
      input_spec=input_spec,
      method=encoder.encode_video_and_text if is_video else None,
      config=config,
      rng=rng,
  )

  train_state = train_utils.TrainState(params=params, model_state=model_state)

  return model, train_state


def evaluate(
    *,
    config: ml_collections.ConfigDict,
    model_cls: type[ImageTextModel],
    dataset: dataset_utils.Dataset,
    rng: Optional[jnp.ndarray] = None,
    workdir: Optional[str] = None,  # pylint: disable=unused-argument
    writer: metric_writers.MetricWriter,
) -> Mapping[str, float]:
  """Evaluates a model on zero-shot text-based video classification."""
  input_spec = get_input_spec(
      dataset_meta_data=dataset.meta_data,
      dataset_configs=config.get('dataset_configs', {}),
      train=False)

  is_video = is_video_input(input_spec)

  hashable_config = config.copy_and_resolve_references()
  hashable_config = ml_collections.FrozenConfigDict(hashable_config)

  # Note that different calls of `_replace` with the same contents will yield
  # the same hash.
  dataset = dataclasses.replace(  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
      dataset, meta_data=flax.core.freeze(dataset.meta_data))
  model, train_state = _create_model_and_train_state(
      config=hashable_config,
      dataset=dataset,
      model_cls=model_cls,
      input_spec=input_spec,
      is_video=is_video,
      rng=rng)

  if config.checkpoint:
    train_state = train_utils.restore_checkpoint(workdir, train_state)[0]

  train_state = jax_utils.replicate(train_state)

  dataset_configs = config.get('dataset_configs', {})

  tokenizer_config = dataset_configs.get('tokenizer', {})
  tokenizer = util.create_tokenizer(tokenizer_config)
  tokenizer.initialize()

  class_names = dataset.meta_data.get(
      'classes') or dataset_configs['class_names']
  class_templates = config.get('class_templates', ['{}'])
  classes_with_templates = []
  for class_ in class_names:
    for template in class_templates:
      classes_with_templates.append(template.format(class_))

  tokenized_classes = tokenizer.string_tensor_to_indices(
      classes_with_templates,
      prepend_bos=tokenizer_config.get('prepend_bos', False),
      append_eos=tokenizer_config.get('append_eos', False),
      max_num_tokens=dataset_configs.get('max_num_words', 32),
  )
  tokenized_classes = tokenized_classes._numpy()  # pylint: disable=protected-access
  tokenized_classes = tokenized_classes.reshape(-1, len(class_templates),
                                                tokenized_classes.shape[-1])

  mask = compute_mask(tokenized_classes, config)

  # We pmap here, to avoid OOM by optimizations. Also, it's best to pmap
  # everything here, in case this is run in an eval job. If not, still pmapping
  # it shouldn't take long.
  logging.info('Encoding the classes as text…')

  tokenized_classes, batch_padding_size = pad_array_to_be_divisible(
      tokenized_classes, jax.local_device_count())
  mask, batch_padding_size = pad_array_to_be_divisible(mask,
                                                       jax.local_device_count())

  tokenized_classes, mask = dataset_utils.shard((tokenized_classes, mask))

  compute_class_text_embeddings_pmapped = jax.pmap(
      partial_with_cache(
          compute_class_text_embeddings,
          model=model,
          batch_size=config.get(
              'class_batch_size',
              config.get('eval_batch_size', config.batch_size)),
          debug=hashable_config.get('debug_eval'),
      ),
      axis_name=NUM_DEVICES_AXIS_NAME,
      donate_argnums=(1, 2),
  )

  encoded_classes = jax_utils.unreplicate(
      compute_class_text_embeddings_pmapped(train_state, tokenized_classes,
                                            mask))

  encoded_classes = encoded_classes[:len(encoded_classes) - batch_padding_size]

  logging.info('Classes encoded.')

  eval_step_pmapped = jax.pmap(
      # This function would fail to cache because `encoded_classes` is a
      # `DeviceArray`. Besides, this value would change for different params.
      # So we can't cache it.
      functools.partial(
          eval_step,
          model=model,
          encoded_classes=encoded_classes,
          is_video=is_video,
          debug=config.get('debug_eval'),
      ),
      axis_name=NUM_DEVICES_AXIS_NAME,
      donate_argnums=(1, 2, 3),
  )

  total_steps = get_epoch_steps(config, dataset, split='eval')

  eval_metrics_all = []

  for step, batch in zip(
      tqdm(range(total_steps), desc='Evaluating'), dataset.valid_iter):
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      eval_metrics_batch = eval_step_pmapped(train_state, batch['inputs'],
                                             batch['label'],
                                             batch['batch_mask'])
      eval_metrics_all.append(eval_metrics_batch)

  return log_eval_summary(
      writer=writer,
      eval_metrics=train_utils.unreplicate_and_get(eval_metrics_all),
      step=jax_utils.unreplicate(train_state.global_step),
      prefix=config.get('writer_prefix', 'valid'))
