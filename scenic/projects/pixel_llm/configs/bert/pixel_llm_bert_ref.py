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

# pylint: disable=line-too-long
r"""Default configs for fine-tuning referring expression localization on RefCOCO wiht BERT.

"""

import ml_collections

from scenic.projects.pixel_llm.configs import common


def get_coco_ref_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, *, mask_on=False, use_text_as_context=True, refexp_field='sent', dataset_name='refcoco_unc', weight=1.0):
  """Returns the RefCOCO/RefCOCOg/RefCOCO+/ train source."""

  min_scale = 0.75
  max_scale = 1.25

  context_prefix = 'where is '
  append_eos = True  # Following BLIP2 to append an eos after prompts

  preproc_spec_train = (
      f"parse_ref_coco(refexp_field='{refexp_field}')"
      f"|decode_ref_coco_annotations('{tokenizer_path}', num_captions_per_sample={max_boxes}, max_text_tokens={max_text_tokens}, use_text_as_context={use_text_as_context}, max_context_tokens={max_context_tokens}, append_context_eos={append_eos}, context_prefix='{context_prefix}', refexp_field='{refexp_field}')"
      f'|random_ratio_resize({min_scale}, {max_scale}, {crop_size})'
      f'|fixed_size_crop({crop_size})'
      '|init_padding_mask'
      f'|pad_images({crop_size}, {crop_size})'
      f'|pad_detection_annotations({max_boxes})'
      f"|add_prompt_boxes"
      f"|drop_nested('label', ['refexp_ids'])"
      f"|add_task_mask(['point'])"
  )
  if mask_on:
    preproc_spec_train += f'|pad_masks({max_boxes}, {crop_size}, {crop_size})'

  context_features = {
      'image': {
          'feature_type': 'FixedLen',
          'shape': [],
          'dtype': 'string',
      },
      'image/id': {'feature_type': 'FixedLen', 'dtype': 'int64'},
      'objects/id': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/area': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/bbox': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'objects/label': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/gt_box_index': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/refexp/refexp_id/ragged_row_lengths_0': {
          'feature_type': 'VarLen',
          'dtype': 'int64',
      },
      'objects/refexp/refexp_id/ragged_flat_values': {
          'feature_type': 'VarLen',
          'dtype': 'int64',
      },
      f'objects/refexp/{refexp_field}/ragged_row_lengths_0': {
          'feature_type': 'VarLen',
          'dtype': 'int64',
      },
      f'objects/refexp/{refexp_field}/ragged_flat_values': {
          'feature_type': 'VarLen',
          'dtype': 'string',
      },
  }
  if mask_on:
    context_features['objects/mask'] = {'feature_type': 'VarLen', 'dtype': 'string'}

  tfrecord_meta = {
      'merge_coco_img_safe': {
          'tfrecords': common.MERGE_COCO_IMAGE_SAFE_TRAIN.path,
          'size': common.MERGE_COCO_IMAGE_SAFE_TRAIN.size,
      },
      'refcoco_unc': {
          'tfrecords': common.REFCOCO_UNC_TRAIN.path,
          'size': common.REFCOCO_UNC_TRAIN.size,
      },
      'refcocog_umd': {
          'tfrecords': common.REFCOCOG_UMD_TRAIN.path,
          'size': common.REFCOCOG_UMD_TRAIN.size,
      },
      'refcocoplus_unc': {
          'tfrecords': common.REFCOCOPLUS_UNC_TRAIN.path,
          'size': common.REFCOCOPLUS_UNC_TRAIN.size,
      },
      'uni_mixed_coco': {
          'tfrecords': common.UNI_MIXED_COCO_TRAIN.path,
          'size': common.UNI_MIXED_COCO_TRAIN.size,
      },
      'uni_mixed_vg': {
          'tfrecords': common.UNI_MIXED_VG_TRAIN.path,
          'size': common.UNI_MIXED_VG_TRAIN.size,
      },
      'uni_flickr': {
          'tfrecords': common.UNI_FLICKR_TRAIN.path,
          'size': common.UNI_FLICKR_TRAIN.size,
      },
  }

  split = 'train'

  source = ml_collections.ConfigDict({
      'source': 'tfrecord',  # `tfds` or `dmvr` or `grain`
      'tfrecords': tfrecord_meta[dataset_name]['tfrecords'],
      'size': tfrecord_meta[dataset_name]['size'],
      'context_features': context_features,
      'name': f'{dataset_name}_{split}',
      'shuffle_buffer_size': 10_000,
      'cache': False,
      'preproc_spec': preproc_spec_train,
      'weight': weight,
  })
  return source


def get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, *, mask_on=False, use_text_as_context=True, refexp_field='sent', dataset_name='refcoco_unc', split='validation'):
  """Returns the RefCOCO/RefCOCOg/RefCOCO+/ eval source."""

  context_prefix = 'where is '
  append_eos = True  # Following BLIP2 to append an eos after prompts

  preproc_spec_eval = (
      f"parse_ref_coco(refexp_field='{refexp_field}')"
      f"|decode_ref_coco_annotations('{tokenizer_path}', max_text_tokens={max_text_tokens}, use_text_as_context={use_text_as_context}, max_context_tokens={max_context_tokens}, append_context_eos={append_eos}, context_prefix='{context_prefix}', refexp_field='{refexp_field}')"
      f'|resize_shorter({crop_size}, {crop_size})'
      '|init_padding_mask'
      f'|pad_images({crop_size}, {crop_size})'
      f'|pad_detection_annotations({max_boxes})'
      f"|add_prompt_boxes"
  )
  if mask_on:
    preproc_spec_eval += f'|pad_masks({max_boxes}, {crop_size}, {crop_size})'

  context_features = {
      'image': {
          'feature_type': 'FixedLen',
          'shape': [],
          'dtype': 'string',
      },
      'image/id': {'feature_type': 'FixedLen', 'dtype': 'int64'},
      'objects/id': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/area': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/bbox': {'feature_type': 'VarLen', 'dtype': 'float32'},
      # 'objects/mask': {'feature_type': 'VarLen', 'dtype': 'string'},
      'objects/label': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/gt_box_index': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'objects/refexp/refexp_id/ragged_row_lengths_0': {
          'feature_type': 'VarLen',
          'dtype': 'int64',
      },
      'objects/refexp/refexp_id/ragged_flat_values': {
          'feature_type': 'VarLen',
          'dtype': 'int64',
      },
      f'objects/refexp/{refexp_field}/ragged_row_lengths_0': {
          'feature_type': 'VarLen',
          'dtype': 'int64',
      },
      f'objects/refexp/{refexp_field}/ragged_flat_values': {
          'feature_type': 'VarLen',
          'dtype': 'string',
      },
  }
  if mask_on:
    context_features['objects/mask'] = {'feature_type': 'VarLen', 'dtype': 'string'}

  tfrecord_meta = {
      'refcoco_unc_validation': {
          'tfrecords': common.REFCOCO_UNC_VALIDATION.path,
          'size': common.REFCOCO_UNC_VALIDATION.size,
      },
      'refcoco_unc_testA': {
          'tfrecords': common.REFCOCO_UNC_TESTA.path,
          'size': common.REFCOCO_UNC_TESTA.size,
      },
      'refcoco_unc_testB': {
          'tfrecords': common.REFCOCO_UNC_TESTB.path,
          'size': common.REFCOCO_UNC_TESTB.size,
      },
      'refcocog_umd_validation': {
          'tfrecords': common.REFCOCOG_UMD_VALIDATION.path,
          'size': common.REFCOCOG_UMD_VALIDATION.size,
      },
      'refcocog_umd_test': {
          'tfrecords': common.REFCOCOG_UMD_TEST.path,
          'size': common.REFCOCOG_UMD_TEST.size,
      },
      'refcocoplus_unc_validation': {
          'tfrecords': common.REFCOCOPLUS_UNC_VALIDATION.path,
          'size': common.REFCOCOPLUS_UNC_VALIDATION.size,
      },
      'refcocoplus_unc_testA': {
          'tfrecords': common.REFCOCOPLUS_UNC_TESTA.path,
          'size': common.REFCOCOPLUS_UNC_TESTA.size,
      },
      'refcocoplus_unc_testB': {
          'tfrecords': common.REFCOCOPLUS_UNC_TESTB.path,
          'size': common.REFCOCOPLUS_UNC_TESTB.size,
      },
  }

  dataset_name_split = f'{dataset_name}_{split}'

  source = ml_collections.ConfigDict({
      'source': 'tfrecord',  # `tfds` or `dmvr` or `grain`
      'tfrecords': tfrecord_meta[dataset_name_split]['tfrecords'],
      'size': tfrecord_meta[dataset_name_split]['size'],
      'context_features': context_features,
      'name': f'{dataset_name}_{split}',
      'split': split,
      'shuffle_buffer_size': 1,
      'cache': False,
      'preproc_spec': preproc_spec_eval,
  })
  return source


def get_config():
  """Returns the configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'pixel_llm_bert_ref'

  # Dataset.
  config.dataset_name = 'custom_flexio'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = ['scenic.projects.pixel_llm.io.ops']

  tokenizer_path = common.BERT_TOKENIZER_PATH
  config.dataset_configs.tokenizer_weight_path = tokenizer_path  # Used in evaluation
  max_text_tokens = 40
  max_context_tokens = -1
  use_text_as_context = True
  max_boxes = 16
  crop_size = 384
  refexp_field = 'sent'

  # Train dataset(s).
  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.merge_sources = True
  config.dataset_configs.train.batch_before_merge = True
  config.dataset_configs.train.sources = [
      get_coco_ref_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='uni_flickr', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      get_coco_ref_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='uni_mixed_coco', refexp_field=refexp_field, weight=5.0, use_text_as_context=use_text_as_context),
      get_coco_ref_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='uni_mixed_vg', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
  ]
  config.dataset_configs.train.postproc_spec = 'drop(["_seed"])'

  # Evaluation dataset(s).
  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.merge_sources = False
  config.dataset_configs.eval.sources = [
      get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcoco_unc', split='validation', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      # get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcoco_unc', split='testA', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      # get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcoco_unc', split='testB', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcocog_umd', split='validation', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      # get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcocog_umd', split='test', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcocoplus_unc', split='validation', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      # get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcocoplus_unc', split='testA', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
      # get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcocoplus_unc', split='testB', refexp_field=refexp_field, use_text_as_context=use_text_as_context),
  ]
  config.dataset_configs.eval.postproc_spec = 'drop(["_seed"])'

  config.dataset_configs.refer_configs = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.eval_step_multiplier = 1.0
  config.dataset_configs.refer_configs.refcoco_unc_validation = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcoco_unc_validation.test_annotation_path = common.REFCOCO_UNC_VALIDATION_ANNOTATION
  config.dataset_configs.refer_configs.refcoco_unc_testA = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcoco_unc_testA.test_annotation_path = common.REFCOCO_UNC_TESTA_ANNOTATION
  config.dataset_configs.refer_configs.refcoco_unc_testB = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcoco_unc_testB.test_annotation_path = common.REFCOCO_UNC_TESTB_ANNOTATION
  config.dataset_configs.refer_configs.refcocoplus_unc_validation = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcocoplus_unc_validation.test_annotation_path = common.REFCOCOPLUS_UNC_VALIDATION_ANNOTATION
  config.dataset_configs.refer_configs.refcocoplus_unc_testA = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcocoplus_unc_testA.test_annotation_path = common.REFCOCOPLUS_UNC_TESTA_ANNOTATION
  config.dataset_configs.refer_configs.refcocoplus_unc_testB = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcocoplus_unc_testB.test_annotation_path = common.REFCOCOPLUS_UNC_TESTB_ANNOTATION
  config.dataset_configs.refer_configs.refcocog_umd_validation = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcocog_umd_validation.test_annotation_path = common.REFCOCOG_UMD_VALIDATION_ANNOTATION
  config.dataset_configs.refer_configs.refcocog_umd_test = ml_collections.ConfigDict()
  config.dataset_configs.refer_configs.refcocog_umd_test.test_annotation_path = common.REFCOCOG_UMD_TEST_ANNOTATION
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
  config.model.git_preprocess_args.image_size = (384, 384)

  config.model.det_backbone_name = 'none'

  config.model.sam_backbone_name = 'sam_vit'
  config.model.sam_backbone_args = ml_collections.ConfigDict()
  config.model.sam_backbone_args.embed_dim = 1280
  config.model.sam_backbone_args.depth = 32
  config.model.sam_backbone_args.num_heads = 16
  config.model.sam_backbone_args.drop_path_rate = 0.5
  config.model.sam_backbone_args.window_block_indexes = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30)
  config.model.sam_preprocess_args = ml_collections.ConfigDict()
  config.model.sam_preprocess_args.image_size = (384, 384)

  config.model.text_decoder_feature_key = 'git_visual_features+sam_visual_features'

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

  config.model.mask_decoder_name = 'none'

  config.model.prompt_fuse_fn = 'sparse'
  config.model.decode_method = 'beam'
  config.model.decode_beam_size = 3
  config.model.decode_per_node_beam_size = 1
  config.model.mult_caption_score = True
  config.model.max_caption_length = max_text_tokens

  config.model.point_predictor_name = 'mlp_point_predictor'
  config.model.gt_box_points_per_side = -1
  config.model.point_loss_type = 'l1_nonzero'
  config.model.use_points_as_det = True
  config.model.point_output_ignore = '^end-1'
  config.model.point_loss_weight = 0.1
  config.model.point_predictor_args = ml_collections.ConfigDict()
  config.model.point_predictor_args.num_output_points = 2
  config.model.point_predictor_args.pre_norm = True
  config.model.point_predictor_args.mlp_activation = 'gelu'
  config.model.point_predictor_args.depth = 4

  config.weights = '/path/to/pixel_llm_bert_trace',
  config.skip_wrong_shape = False

  # Training.
  config.batch_size = 64
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
      )

  iter_factor = 4
  # learning rate and training schedule
  config.num_training_steps = 30_000 * iter_factor
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


