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

# pylint: disable=line-too-long,unused-variable,unused-argument
r"""Default configs for pre-training on COCO caption and Localized Narratives with T5.

"""

import ml_collections

from scenic.projects.pixel_llm.configs import common


def get_ln_trace_cap_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, *, dataset_name='coco', weight=1.0):
  """Returns the localized narrative train source."""
  # min_scale = 0.75
  # max_scale = 1.25
  min_scale = 1.0
  max_scale = 1.0

  prompt = "['A long image caption: ', 'A long image description: ', 'Write a long description for the image. ', 'Write a long description for the photo. ', 'Provide a long description of what is presented in the photo. ', 'Describe the content of the image in detail. ', 'Can you in detail explain what you see in the image? ', 'Could you use a few sentences to describe what you perceive in the photo? ', 'Please provide a long depiction of the picture. ', 'Using language, provide a long account of the image. ', 'Use a few senetences to illustrate what is happening in the picture. ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  num_points_per_token = 2

  preproc_spec_train = (
      f"decode_localized_narratives_annotations('{tokenizer_path}', {max_boxes}, {max_text_tokens}, {num_points_per_token})"
      f'|random_ratio_resize({min_scale}, {max_scale}, {crop_size})'
      f'|fixed_size_crop({crop_size})'
      '|init_padding_mask'
      f'|pad_images({crop_size}, {crop_size})'
      f"|pad_caption_annotations({max_boxes})"
      f'|pad_loco_annotations({max_boxes}, {num_points_per_token})'
      f"|add_prompt_tokens('{tokenizer_path}', {max_boxes}, {prompt}, {max_context_tokens}, {append_eos})"
      '|add_prompt_boxes'
      "|add_task_mask(['point', 'caption'])"
  )

  sequence_features = {
      'caption/utterance': {'feature_type': 'VarLen', 'dtype': 'string'},
      'caption/center': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'caption/bbox': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'caption/point': {'feature_type': 'VarLen', 'dtype': 'float32'},
  }

  context_features = {
      'image/encoded': {
          'feature_type': 'FixedLen',
          'shape': [],
          'dtype': 'string',
      },
      'image/id': {'feature_type': 'VarLen', 'dtype': 'string'},
      'caption/string': {'feature_type': 'VarLen', 'dtype': 'string'},
  }

  tfrecord_meta = {
      'coco': {
          'tfrecords': common.LN_COCO_TRAIN.path,
          'size': common.LN_COCO_TRAIN.size,
      },
  }
  source = ml_collections.ConfigDict({
      'source': 'tfrecord',
      'tfrecords': tfrecord_meta[dataset_name]['tfrecords'],
      'size': tfrecord_meta[dataset_name]['size'],
      'context_features': context_features,
      'sequence_features': sequence_features,
      'shuffle_buffer_size': 10_000,
      'cache': False,
      'preproc_spec': preproc_spec_train,
      'weight': weight,
  })
  return source


def get_ln_trace_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, *, dataset_name='coco'):
  """Returns the localized narrative train source."""

  prompt = "['A long image caption: ']"
  # prompt = "['we can see ']"
  # prompt = "['where is ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  num_points_per_token = 2

  preproc_spec_eval = (
      f"decode_localized_narratives_annotations('{tokenizer_path}', {max_boxes}, {max_text_tokens}, {num_points_per_token}, with_image_id=True)"
      f"|resize_shorter({crop_size}, {crop_size})"
      f"|init_padding_mask"
      f"|pad_images({crop_size}, {crop_size})"
      f"|pad_caption_annotations({max_boxes})"
      f'|pad_loco_annotations({max_boxes}, {num_points_per_token})'
      f"|add_prompt_tokens('{tokenizer_path}', {max_boxes}, {prompt}, {max_context_tokens}, {append_eos})"
      f"|add_prompt_boxes"
  )
  sequence_features = {
      'caption/utterance': {'feature_type': 'VarLen', 'dtype': 'string'},
      'caption/center': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'caption/bbox': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'caption/point': {'feature_type': 'VarLen', 'dtype': 'float32'},
  }

  context_features = {
      'image/encoded': {
          'feature_type': 'FixedLen',
          'shape': [],
          'dtype': 'string',
      },
      'image/id': {'feature_type': 'VarLen', 'dtype': 'string'},
      'caption/string': {'feature_type': 'VarLen', 'dtype': 'string'},
  }

  tfrecord_meta = {
      'coco': {
          'tfrecords': common.LN_COCO_VAL.path,
          'size': common.LN_COCO_VAL.size,
      }
  }

  source = ml_collections.ConfigDict({
      'source': 'tfrecord',
      'tfrecords': tfrecord_meta[dataset_name]['tfrecords'],
      'size': tfrecord_meta[dataset_name]['size'],
      'context_features': context_features,
      'sequence_features': sequence_features,
      'shuffle_buffer_size': 1,
      'cache': False,
      'preproc_spec': preproc_spec_eval,
      'name': f'ln_{dataset_name}_trace_val',
  })
  return source


def get_coco_cap_train_source(tokenizer_path, max_text_tokens, max_context_tokens, num_captions_per_sample, crop_size, weight=1.0):
  """Returns the COCO Caption train source."""
  tfds_name = 'coco_captions'

  min_scale = 0.75
  max_scale = 1.25

  if max_text_tokens < 32:
    prompt = "['A photo of ']"
  else:
    prompt = "['A short image caption: ', 'A short image description: ', 'A photo of ', 'An image that shows ', 'Write a short description for the image. ', 'Write a description for the photo. ', 'Provide a description of what is presented in the photo. ', 'Briefly describe the content of the image. ', 'Can you briefly explain what you see in the image? ', 'Could you use a few words to describe what you perceive in the photo? ', 'Please provide a short depiction of the picture. ', 'Using language, provide a short account of the image. ', 'Use a few words to illustrate what is happening in the picture. ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  num_points_per_token = 2

  preproc_spec_train = (
      f"decode_coco_caption_annotations('{tokenizer_path}', {num_captions_per_sample}, {max_text_tokens})"
      f"|random_ratio_resize({min_scale}, {max_scale}, {crop_size})"
      f"|fixed_size_crop({crop_size})"
      f"|init_padding_mask"
      f"|pad_images({crop_size}, {crop_size})"
      f"|pad_caption_annotations({num_captions_per_sample})"
      f'|pad_loco_annotations({num_captions_per_sample}, {num_points_per_token})'
      f"|add_prompt_tokens('{tokenizer_path}', {num_captions_per_sample}, {prompt}, {max_context_tokens}, {append_eos})"
      f"|add_prompt_boxes"
      f"|add_task_mask(['caption'])"
  )
  source = ml_collections.ConfigDict({
      'source': 'tfds',  # `tfds` or `dmvr`
      'tfds_name': tfds_name,
      'split': 'train+restval',
      'shuffle_buffer_size': 10_000,
      'cache': False,
      'preproc_spec': preproc_spec_train,
      'weight': weight,
    })
  return source


def get_coco_cap_eval_source(
    tokenizer_path,
    max_text_tokens,
    max_context_tokens,
    num_captions_per_sample,
    crop_size,
):
  """Returns the COCO Caption eval source."""
  tfds_name = 'coco_captions'

  prompt = "['A photo of ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  preproc_spec_eval = (
      f"decode_coco_caption_annotations('{tokenizer_path}', {num_captions_per_sample}, {max_text_tokens})"
      f"|resize_shorter({crop_size})"
      f"|center_crop({crop_size})"
      f"|init_padding_mask"
      f"|pad_images({crop_size}, {crop_size})"
      f"|add_prompt_tokens('{tokenizer_path}', 1, {prompt}, {max_context_tokens}, {append_eos})"
      f"|add_prompt_boxes(num_prompts=1)"
  )

  source = ml_collections.ConfigDict({
      'source': 'tfds',
      'tfds_name': tfds_name,
      'name': 'coco_captions_val',
      'split': 'val',
      'shuffle_buffer_size': 1,
      'cache': False,
      'preproc_spec': preproc_spec_eval,
  })

  return source


def get_config():
  """Returns the configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'pixel_llm_t5_trace'

  # Dataset.
  config.dataset_name = 'custom_flexio'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = ['scenic.projects.pixel_llm.io.ops']

  tokenizer_path = 't5'
  config.dataset_configs.tokenizer_weight_path = tokenizer_path  # Used in evaluation
  max_text_tokens = 128
  max_context_tokens = 8
  max_boxes = 1
  crop_size = 384

  # Train dataset(s).
  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.merge_sources = True
  config.dataset_configs.train.batch_before_merge = True
  config.dataset_configs.train.sources = [
      get_coco_cap_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, weight=1.0),
      get_ln_trace_cap_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='coco', weight=2.0),
  ]
  config.dataset_configs.train.postproc_spec = 'drop(["_seed"])'

  # Evaluation dataset(s).
  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.merge_sources = False
  config.dataset_configs.eval.sources = [
      get_coco_cap_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size),
      get_ln_trace_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, 1, crop_size, dataset_name='coco'),
  ]
  config.dataset_configs.eval.postproc_spec = 'drop(["_seed"])'

  # Dataset configs needed by the trainer.
  config.dataset_configs.caption_configs = ml_collections.ConfigDict()
  config.dataset_configs.caption_configs.test_annotation_path = common.COCO_CAP_TEST_ANNOTATION

  config.dataset_configs.extra_meta_data = {
      'input_shape': [-1, crop_size, crop_size, 3],
      'prompt_box_shape': [-1, 1, 4],
      }

  config.rng_seed = 0

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'pixel_llm'

  config.model.git_backbone_name = 'eva02_vit'
  config.model.git_backbone_args = ml_collections.ConfigDict()
  config.model.git_backbone_args.embed_dim = 1024
  config.model.git_backbone_args.depth = 24
  config.model.git_backbone_args.num_heads = 16
  config.model.git_backbone_args.patch_size = 14
  config.model.git_backbone_args.mlp_ratio = 4 * 2 / 3
  config.model.git_backbone_args.use_ln_post = True
  config.model.git_backbone_args.drop_path_rate = 0.1
  config.model.git_preprocess_args = ml_collections.ConfigDict()
  config.model.git_preprocess_args.image_size = (crop_size, crop_size)

  config.model.det_backbone_name = 'none'

  config.model.sam_backbone_name = 'sam_vit'
  config.model.sam_backbone_args = ml_collections.ConfigDict()
  config.model.sam_backbone_args.embed_dim = 1280
  config.model.sam_backbone_args.depth = 32
  config.model.sam_backbone_args.num_heads = 16
  config.model.sam_backbone_args.drop_path_rate = 0.5
  config.model.sam_backbone_args.window_block_indexes = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30)
  config.model.sam_preprocess_args = ml_collections.ConfigDict()
  config.model.sam_preprocess_args.image_size = (crop_size, crop_size)

  config.model.text_decoder_feature_key = 'git_visual_features+sam_visual_features'
  # T5 setting
  config.model.vocab_size = common.SP_VOCAB_SIZE
  config.model.begin_token_id = 0
  config.model.end_token_id = 1
  config.model.text_decoder_name = 'flan_t5_xl'
  config.model.text_decoder_args = ml_collections.ConfigDict()
  config.model.text_decoder_args.dtype = 'float32'  # 'bfloat16'
  config.model.text_decoder_args.dropout_rate = 0.1
  config.model.text_decoder_args.encoder_lora_rank = 32
  config.model.text_decoder_args.decoder_lora_rank = 32
  config.model.text_decoder_args.encoder_lora_scale = 2
  config.model.text_decoder_args.decoder_lora_scale = 2
  config.model.text_decoder_args.encoder_lora_modules = 'q,v'
  config.model.text_decoder_args.decoder_lora_modules = 'q,v'

  config.model.box_decoder_name = 'none'

  config.model.prompt_drop_rate = 0.0
  config.model.prompt_use_box_rate = 1.0
  config.model.prompt_encoder_name = 'sam_prompt_encoder'
  config.model.prompt_encoder_args = ml_collections.ConfigDict()
  config.model.prompt_encoder_args.embed_dim = 256
  config.model.prompt_adapter_name = 'sam_prompt_adapter'
  config.model.prompt_adapter_args = ml_collections.ConfigDict()
  config.model.prompt_adapter_args.depth = 6
  config.model.prompt_adapter_args.transformer_dim = 512
  config.model.prompt_adapter_args.output_dim = 1024

  config.model.visual_project_layers_name = 'linear'
  config.model.visual_project_layers_args = ml_collections.ConfigDict()
  config.model.visual_project_layers_args.emb_dim = 2048

  config.model.mask_decoder_name = 'none'

  config.model.prompt_fuse_fn = 'sparse'
  config.model.decode_method = 'beam'
  config.model.decode_beam_size = 4
  config.model.decode_per_node_beam_size = 2
  # config.model.decode_method = 'greedy'
  # config.model.decode_beam_size = 1
  config.model.max_caption_length = max_text_tokens

  config.model.point_predictor_name = 'mlp_point_predictor'
  config.model.gt_box_points_per_side = -1
  config.model.point_loss_type = 'l1_nonzero'
  config.model.use_points_as_det = True
  config.model.point_output_ignore = 'begin,end,pad'
  config.model.trace_point_output_ignore = 'begin,end,pad'
  config.model.point_loss_weight = 0.1
  config.model.point_predictor_args = ml_collections.ConfigDict()
  config.model.point_predictor_args.num_output_points = 2
  config.model.point_predictor_args.pre_norm = True
  config.model.point_predictor_args.mlp_activation = 'gelu'
  config.model.point_predictor_args.depth = 4

  config.weights='/path/to/pixel_llm_t5_webli',
  config.load_pretrained_t5_weights = False
  config.skip_wrong_shape = False
  config.load_prefix = ''

  # Training.
  config.batch_size = 256
  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.0
  config.optimizer.skip_scale_and_bias_regularization = True
  config.frozen_params = (
      ('.*sam_backbone.*', 'sam_backbone'),
      ('.*prompt_encoder.*', 'prompt_encoder'),
      # ('.*git_backbone.*', 'git_backbone'),
      ('.*mask_decoder.*', 'mask_decoder'),
      # ('.*t5_module.*', 'T5'),
      ('^(?!.*lora.*).*t5_module.*', 'T5'),
      )

  iter_factor = 1
  # learning rate and training schedule
  config.num_training_steps = 10_000 * iter_factor
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.warmup_steps = 250 * iter_factor
  config.lr_configs.base_learning_rate = 5e-5

  config.checkpoint_steps = 1000 * iter_factor
  config.log_eval_steps = 2500 * iter_factor

  # Logging.
  config.eval_meteor_spice = False
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 50  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


