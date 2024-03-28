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

"""Scenic trainer for zero-shot text-to-visual retrieval evaluation.

Note this "trainer" doesn't actually train but just evaluates.
"""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import dataclasses
from typing import Optional

from clu import metric_writers
from flax import jax_utils
import flax.core
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.projects.lang4video.model import loss
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
from scenic.projects.lang4video.trainer.train_utils import pad_and_batch
from scenic.projects.lang4video.trainer.train_utils import partial_with_cache
from scenic.train_lib import train_utils
from tqdm.auto import tqdm


# TODO(sacastro): support multiple clips.


# We pass the model and not directly the encoder because it's hashable.
def eval_step(
    train_state: train_utils.TrainState,
    visual: jnp.ndarray,  # Shape: (N, F, H, W, C) if `is_video`, w/o F if not.
    text: jnp.ndarray,  # Shape: (N, L)
    mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
    *,
    model: ImageTextModel,
    is_video: bool,
    debug: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:  # Shapes: (N, E), (N, E)
  """Runs a single step of evaluation."""
  encoded_visual, encoded_text = model.flax_model.apply(
      variables={
          'params': train_state.params,
          **train_state.model_state,
      },
      method=model.flax_model.encode_video_and_text if is_video else None,
      **{'video' if is_video else 'image': visual},
      text=text,
      mask=mask,
      train=False,
      debug=debug)

  if axis_name_exists(NUM_DEVICES_AXIS_NAME):
    encoded_visual = jax.lax.all_gather(encoded_visual, NUM_DEVICES_AXIS_NAME)
    encoded_visual = encoded_visual.reshape(-1, encoded_visual.shape[-1])

    encoded_text = jax.lax.all_gather(encoded_text, NUM_DEVICES_AXIS_NAME)
    encoded_text = encoded_text.reshape(-1, encoded_text.shape[-1])

  return encoded_visual, encoded_text


def compute_retrieval_metrics(
    scores: jnp.ndarray,  # Shape: (N, M)
    target: Optional[jnp.ndarray] = None,  # Shape: (N,)
    k_values: Sequence[int] = (1, 5, 10),
    suffix: str = '',
    suffix_separator: str = '_',
) -> MutableMapping[str, float]:
  """Computes the text-to-visual retrieval metrics."""
  if target is None:
    assert scores.shape[0] == scores.shape[1], ('If the score matrix is not a'
                                                ' square, the target values'
                                                ' need to be provided as they'
                                                " can't be easily inferred.")
    target = jnp.arange(scores.shape[0])

  target = target.reshape(-1, 1)

  if suffix:
    suffix = suffix_separator + suffix

  summary = {}

  sorted_score_positions = scores.argsort(axis=-1)[..., ::-1]
  target_position_mask = sorted_score_positions == target

  for k in k_values:
    summary[f'recall@{k}{suffix}'] = target_position_mask[:, :k].any(
        axis=-1).mean()

  # We specify `size` so this function can be jitted.
  ranks = jnp.argwhere(target_position_mask, size=len(scores))[:, -1] + 1
  summary[f'ranks{suffix}'] = ranks
  summary[f'median_rank{suffix}'] = jnp.median(ranks)
  summary[f'mean_rank{suffix}'] = ranks.mean()

  return summary  # pytype: disable=bad-return-type  # jax-ndarray


def _compute_metrics(
    encoded_visuals: jnp.ndarray,  # Shape: (N, E)
    encoded_texts: jnp.ndarray,  # Shape: (N, E)
    model: ImageTextModel,
    loss_fn: Callable[..., jnp.ndarray],
    retrieval_batch_size: Optional[int] = None,
) -> MutableMapping[str, float]:
  """Computes the similarity and the metrics."""
  assert encoded_visuals.shape == encoded_texts.shape

  n = encoded_texts.shape[0]

  # We don't batch because of potential OOMs but because we may want to compute
  # the metrics with a different number of negatives.
  retrieval_batch_size = retrieval_batch_size or n

  # We don't use `batch_and_vmap` here because we need to use the batched output
  # with some special padding values for the loss.
  batched, batch_padding_size = pad_and_batch((encoded_texts, encoded_visuals),
                                              batch_size=retrieval_batch_size)
  batched_scores = jax.vmap(model.flax_model.compute_similarity)(*batched)

  n_batches = batched_scores.shape[0]

  # We set the padding columns to -inf, so we can ignore them in the scores.
  batched_scores = batched_scores.at[-1, :, retrieval_batch_size -
                                     batch_padding_size:].set(-jnp.inf)

  scores = batched_scores.reshape(-1, retrieval_batch_size)[:n]
  target = jnp.tile(jnp.arange(retrieval_batch_size), n_batches)[:n]
  metrics_summary = compute_retrieval_metrics(scores, target)

  where = jnp.ones((n_batches, retrieval_batch_size)).astype(jnp.bool_)
  where = where.at[-1, retrieval_batch_size - batch_padding_size:].set(False)

  initial = jnp.full((n_batches,), -jnp.inf)
  metrics_summary['loss'] = jax.vmap(loss_fn)(
      batched_scores, where=where, initial=initial).mean(where=where)

  return metrics_summary  # pytype: disable=bad-return-type  # jax-ndarray


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
  """Evaluates a model on zero-shot text-to-visual retrieval."""
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

  eval_step_pmapped = jax.pmap(
      partial_with_cache(
          eval_step,
          model=model,
          is_video=is_video,
          debug=config.get('debug_eval'),
      ),
      axis_name=NUM_DEVICES_AXIS_NAME,
      donate_argnums=(1, 2, 3),
  )

  total_steps = get_epoch_steps(config, dataset, split='eval')

  encoded_visual_list = []
  encoded_text_list = []

  for step, batch in zip(
      tqdm(range(total_steps), desc='Evaluating'), dataset.valid_iter):
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      batch.pop('key', None)

      if (text := batch.get('text_indices')) is None:
        text = batch['label']
      else:
        text = text.squeeze(axis=-2)

      mask = compute_mask(text, config)

      encoded_visual, encoded_text = eval_step_pmapped(
          train_state,
          batch['inputs'],
          text,
          mask,
      )
      encoded_visual = jax_utils.unreplicate(encoded_visual)
      encoded_text = jax_utils.unreplicate(encoded_text)

      batch_mask = batch['batch_mask'].reshape(-1).astype(bool)

      for mask_instance, encoded_visual_instance, encoded_text_instance in zip(
          batch_mask, encoded_visual, encoded_text):
        if mask_instance:
          encoded_visual_list.append(encoded_visual_instance)
          encoded_text_list.append(encoded_text_instance)

  encoded_visuals = jnp.stack(encoded_visual_list)
  encoded_texts = jnp.stack(encoded_text_list)

  # This evaluation may be done multiple times in an eval job, so we jit here.
  # Still, in case it's run only once, it shouldn't take long to jit, as it's
  # only evaluation.
  metrics_summary = jax.jit(
      partial_with_cache(
          _compute_metrics,
          model=model,
          # This is a workaround for checking if we are evaluating the
          # validation split from the dataset used for training. In that case,
          # we want to compute the loss with the same function as in training.
          # Otherwise, we want a retrieval-like loss.
          loss_fn=model.loss_function
          if 'retrieval_batch_size' in config else loss.nce_loss,
          retrieval_batch_size=config.get('retrieval_batch_size'),
      ),
      backend='cpu',  # To avoid OOM while compiling.
      donate_argnums=(0, 1),
  )(encoded_visuals, encoded_texts)

  step = jax_utils.unreplicate(train_state.global_step)
  prefix = config.get('writer_prefix', 'valid')

  ranks = metrics_summary.pop('ranks')
  if step % (config.get('log_eval_steps', 100) * 10) == 0:
    writer.write_histograms(step, {f'{prefix}/ranks': ranks})

  return log_eval_summary(
      writer=writer,
      extra_eval_summary=metrics_summary,
      step=step,
      prefix=prefix)
