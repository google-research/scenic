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

"""Base config."""

from collections.abc import Sequence
from typing import Literal
from typing import Union

import ml_collections


def _pp_ops(
    train: bool,
    # Visual args:
    visual_key: str = 'image',
    is_video: bool = False,
    num_frames: int = 16,
    stride: int = 14,
    min_resize: int = 256,
    crop_size: Union[int, Sequence[int, int]] = 224,
    resizing_method: str = 'bicubic',
    normalization_mean: Union[Sequence[int], int] = 0,
    normalization_std: Union[Sequence[int], int] = 1,
    # Text args:
    text_in_key: str = 'texts',
    text_out_key: str = 'labels',
    is_classification: bool = False,
    tokenizer_type: Literal['dmvr', 'sentence_piece', ''] = '',
    tokenizer_name: str = '',  # Only used for the DMVR tokenizers.
    vocab_name_or_path: str = '',  # Only used for SentencePiece tokenizers.
    prepend_bos: bool = False,  # Only used for the DMVR tokenizers.
    append_eos: bool = True,
    lowercase: bool = False,  # Only used for SentencePiece tokenizers.
    max_num_tokens: int = 77,
    pad_value: int = 0,  # Only used for SentencePiece tokenizers.
) -> str:
  """Returns the pre-processing operations string."""
  if is_video:
    # TODO(sacastro): support multiple clips.
    sample_step = (f'|sample_sequence(num_steps={1 if train else num_frames},'
                   f'                 random={train}, stride={stride},'
                   f'                 inkey="{visual_key}",'
                   f'                 outkey="{visual_key}")')
  else:
    sample_step = ''

  decode_fn_name = 'decode_frames' if is_video else 'decode'

  if not is_classification and tokenizer_type:
    text_sampling_fn = 'random_sample' if train else 'first'
    text_sampling_step = (f'|{text_sampling_fn}(inkey="{text_in_key}",'
                          f'                    outkey="{text_out_key}")')
  else:
    text_sampling_step = ''

  if tokenizer_type == 'dmvr':
    # assert not lowercase  # This arg is used in other places.
    assert pad_value == 0
    vocab_name_or_path_str = (f'"{vocab_name_or_path}"'
                              if vocab_name_or_path else None)
    tokenization_step = (f'|dmvr_tokenize("{tokenizer_name}",'
                         f'               vocab_path={vocab_name_or_path_str},'
                         f'               prepend_bos={prepend_bos},'
                         f'               append_eos={append_eos},'
                         f'               max_num_tokens={max_num_tokens},'
                         f'               inkey="{text_out_key}",'
                         f'               outkey="{text_out_key}")')
  elif tokenizer_type == 'sentence_piece':
    assert not prepend_bos
    tokenization_step = (f'|tokenize(max_len={max_num_tokens},'
                         f'          eos="{"yes" if append_eos else "none"}",'
                         f'          model="{vocab_name_or_path}",'
                         f'          lower={lowercase},'
                         f'          sample_if_multi={train},'
                         f'          pad_value={pad_value},'
                         f'          inkey="{text_out_key}",'
                         f'          outkey="{text_out_key}")')
  else:
    tokenization_step = ''

  return (f'{sample_step}'
          f'|{decode_fn_name}(inkey="{visual_key}", outkey="{visual_key}")'
          f'|resize_smallest({min_resize}, method="{resizing_method}",'
          f'                 inkey="{visual_key}", outkey="{visual_key}")'
          f'|{"random" if train else "central"}_crop({crop_size},'
          f'                                         inkey="{visual_key}",'
          f'                                         outkey="{visual_key}")'
          f'|value_range(0, 1, inkey="{visual_key}", outkey="{visual_key}")'
          f'|standardize(mean={normalization_mean}, std={normalization_std},'
          f'             inkey="{visual_key}", outkey="{visual_key}")'
          f'{text_sampling_step}'
          f'{tokenization_step}'
          f'|keep("{visual_key}", "{text_out_key}")').strip('|')


def get_train_ops(*args, **kwargs) -> str:
  return _pp_ops(train=True, *args, **kwargs)


def get_eval_ops(*args, **kwargs) -> str:
  return _pp_ops(train=False, *args, **kwargs)


def _metadata(
    vocab_size: int,
    max_num_tokens: int,
) -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict({
      'vocab_size': vocab_size,
      'target_shape': (-1, max_num_tokens),
  })


def _batch_size_per_device(
    batch_size_per_device_memory_gb: int,
    device_memory_gbs: int,
) -> int:
  return batch_size_per_device_memory_gb * device_memory_gbs


def _batch_size(
    batch_size_per_device: int,
    device_count: int,
) -> int:
  return batch_size_per_device * device_count


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  run_local = bool(run_local)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'lang4video'
  config.rng_seed = 0

  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = ''

  config.batch_size_per_device_memory_gb = 16
  config.device_memory_gbs = 16

  if run_local:
    config.batch_size_per_device = 8
  else:
    config.batch_size_per_device = _batch_size_per_device(
        batch_size_per_device_memory_gb=config.get_ref(
            'batch_size_per_device_memory_gb'),
        device_memory_gbs=config.get_ref('device_memory_gbs'))
  config.device_count = 1

  config.batch_size = _batch_size(
      batch_size_per_device=config.get_ref('batch_size_per_device'),
      device_count=config.get_ref('device_count'))
  config.eval_batch_size = config.get_ref('batch_size')

  config.steps_per_eval = 10 if run_local else 0

  # We set different dtypes for the data and the model because we may want a
  # mix.
  #
  # We leave it empty, so we use the default values from the models.
  config.model_dtype_str = ''

  config.model = ml_collections.ConfigDict()
  config.model.load_pretrained_vars = True
  config.model.loss = 'nce'
  config.model.similarity = 'cosine'
  config.model.gather_scores = True

  config.model.encoder = ml_collections.ConfigDict()

  config.data_dtype_str = 'float32'

  config.retrieval_batch_size = 0  # For the retrieval evaluation.

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_host = 'autotune'
  config.dataset_configs.prefetch_to_device = 2

  config.dataset_configs.readahead = True

  config.dataset_configs.load_train = True
  config.dataset_configs.load_val = True
  config.dataset_configs.load_test = False

  # The following configurations are meant to be used by the DMVR datasets but
  # also as input to other parts of the config.

  # Given fps = 25, this means we are roughly sampling 1 frame per second.
  config.dataset_configs.num_frames = 16
  config.dataset_configs.stride = 14
  config.dataset_configs.zero_centering = False
  config.dataset_configs.resize_method = 'bicubic'
  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.normalization_mean = (0, 0, 0)
  config.dataset_configs.normalization_std = (1, 1, 1)
  config.dataset_configs.tokenizer = ml_collections.ConfigDict()
  config.dataset_configs.tokenizer.tokenizer_type = ''
  config.dataset_configs.tokenizer.tokenizer_vocab = ''
  config.dataset_configs.tokenizer.prepend_bos = True
  config.dataset_configs.tokenizer.append_eos = True
  config.dataset_configs.max_num_words = 77

  # The following configurations are not used by the DMVR datasets, but we
  # control them for here.

  config.dataset_configs.visual_key = 'image'
  config.dataset_configs.is_video = False
  config.dataset_configs.text_in_key = 'texts'
  config.dataset_configs.text_out_key = 'labels'
  config.dataset_configs.is_classification = False
  config.dataset_configs.tokenizer_type = ''
  config.dataset_configs.vocab_size = -2
  config.dataset_configs.lowercase = False
  config.dataset_configs.pad_value = 0

  # For the datasets from Big Transfer (that use pp ops from):
  # (note we can't put these params in the mixin because the refs don't exist)

  config.dataset_configs.feature_key = config.dataset_configs.get_ref(
      'visual_key')
  ops_kwargs = {
      'visual_key':
          config.dataset_configs.visual_key,
      'is_video':
          config.dataset_configs.is_video,
      'num_frames':
          config.dataset_configs.num_frames,
      'stride':
          config.dataset_configs.stride,
      'min_resize':
          config.dataset_configs.min_resize,
      'crop_size':
          config.dataset_configs.crop_size,
      'normalization_mean':
          config.dataset_configs.normalization_mean,
      'normalization_std':
          config.dataset_configs.normalization_std,
      'text_in_key':
          config.dataset_configs.text_in_key,
      'text_out_key':
          config.dataset_configs.text_out_key,
      'tokenizer_type':
          config.dataset_configs.tokenizer_type,
      'is_classification':
          config.dataset_configs.is_classification,
      'tokenizer_name':
          config.dataset_configs.tokenizer.tokenizer_type,
      'vocab_name_or_path':
          config.dataset_configs.tokenizer.tokenizer_vocab,
      'prepend_bos':
          config.dataset_configs.tokenizer.prepend_bos,
      'append_eos':
          config.dataset_configs.tokenizer.append_eos,
      'lowercase':
          config.dataset_configs.lowercase,
      'max_num_tokens':
          config.dataset_configs.max_num_words,
      'pad_value':
          config.dataset_configs.pad_value,
  }
  config.dataset_configs.pp_train = get_train_ops(**ops_kwargs)
  config.dataset_configs.pp_eval = get_eval_ops(**ops_kwargs)
  config.dataset_configs.shuffle_buffer_size = 5 if run_local else 250_000
  config.dataset_configs.num_classes = -1  # To avoid a warning message.
  config.dataset_configs.extra_meta_data = _metadata(
      vocab_size=config.dataset_configs.get_ref('vocab_size'),
      max_num_tokens=config.dataset_configs.get_ref('max_num_words'))

  config.checkpoint = True  # Whether to load a checkpoint if available.
  config.overwrite_checkpoint = run_local
  config.debug_train = False
  config.debug_eval = False
  config.enable_pmap_and_jit = True
  config.use_jax_compilation_cache = False
  if run_local:
    config.count_flops = False

  return config
