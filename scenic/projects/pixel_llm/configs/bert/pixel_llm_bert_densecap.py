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
r"""Default configs for fine-tuning dense object captioning on Visual Genome wiht BERT.

"""

import ml_collections

from scenic.projects.pixel_llm.configs import common


def get_vg_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, *, weight=1.0):
  """Returns the visual genome train source."""

  min_scale = 0.1
  max_scale = 2.0

  prompt = "['an object of ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  preproc_spec_train = (
      'parse_vg'
      f"|decode_vg_annotations('{tokenizer_path}', {max_text_tokens})"
      f"|random_horizontal_flip"
      f"|random_ratio_resize({min_scale}, {max_scale}, {crop_size})"
      f"|fixed_size_crop({crop_size})"
      f"|init_padding_mask"
      f"|pad_images({crop_size}, {crop_size})"
      f"|pad_detection_annotations({max_boxes})"
      f"|add_prompt_tokens('{tokenizer_path}', {max_boxes}, {prompt}, {max_context_tokens}, {append_eos})"
  )

  context_features = {
      'image': {
          'feature_type': 'FixedLen',
          'shape': [],
          'dtype': 'string',
      },
      'img_id': {'feature_type': 'FixedLen', 'dtype': 'int64'},
      'regions/id': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'regions/bbox': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'regions/phrase': {'feature_type': 'VarLen', 'dtype': 'string'},
  }

  tfrecord_meta = {
      'vg_densecap': {
          'tfrecords': common.VG_TRAIN.path,
          'size': common.VG_TRAIN.size,
      }
  }

  dataset_name = 'vg_densecap'
  split = 'train'

  source = ml_collections.ConfigDict({
      'source': 'tfrecord',
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


def get_vg_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='vg', split='test', task='densecap'):
  """Returns the visual genome train source."""

  prompt = "['an object of ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  preproc_spec_eval = (
      'parse_vg'
      f"|decode_vg_annotations('{tokenizer_path}', {max_text_tokens})"
      f"|resize_shorter({crop_size}, {crop_size})"
      f"|init_padding_mask"
      f"|pad_images({crop_size}, {crop_size})"
      f"|pad_detection_annotations({max_boxes})"
      f"|add_prompt_tokens('{tokenizer_path}', {max_boxes}, {prompt}, {max_context_tokens}, {append_eos})"
  )

  context_features = {
      'image': {
          'feature_type': 'FixedLen',
          'shape': [],
          'dtype': 'string',
      },
      'img_id': {'feature_type': 'FixedLen', 'dtype': 'int64'},
      'regions/id': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'regions/bbox': {'feature_type': 'VarLen', 'dtype': 'float32'},
      'regions/phrase': {'feature_type': 'VarLen', 'dtype': 'string'},
  }

  tfrecord_meta = {
      'vg': {
          'tfrecords': common.VG_TEST.path,
          'size': common.VG_TEST.size,
      }
  }

  source = ml_collections.ConfigDict({
      'source': 'tfrecord',
      'tfrecords': tfrecord_meta[dataset_name]['tfrecords'],
      'size': tfrecord_meta[dataset_name]['size'],
      'context_features': context_features,
      'name': f'{dataset_name}_{task}_{split}',
      'shuffle_buffer_size': 1,
      'cache': False,
      'preproc_spec': preproc_spec_eval,
  })
  return source


def get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, *, mask_on=False, refexp_field='sent', dataset_name='refcoco_unc', split='validation'):
  """Returns the RefCOCO/RefCOCOg/RefCOCO+/ train source."""

  prompt = "['an object of ']"
  append_eos = True  # Following BLIP2 to append an eos after prompts

  preproc_spec_eval = (
      f"parse_ref_coco(refexp_field='{refexp_field}')"
      f"|decode_ref_coco_annotations('{tokenizer_path}', {max_text_tokens}, refexp_field='{refexp_field}')"
      f'|resize_shorter({crop_size}, {crop_size})'
      '|init_padding_mask'
      f'|pad_images({crop_size}, {crop_size})'
      f'|pad_detection_annotations({max_boxes})'
      f"|add_prompt_tokens('{tokenizer_path}', {max_boxes}, {prompt}, {max_context_tokens}, {append_eos})"
      f"|drop_nested('label', ['refexp_ids'])"
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
      'source': 'tfrecord',
      'tfrecords': tfrecord_meta[dataset_name_split]['tfrecords'],
      'size': tfrecord_meta[dataset_name_split]['size'],
      'context_features': context_features,
      'name': f'{dataset_name}_loca_{split}',
      'split': split,
      'shuffle_buffer_size': 1,
      'cache': False,
      'preproc_spec': preproc_spec_eval,
  })
  return source


def get_config():
  """Returns the configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'pixel_llm_bert_densecap'

  # Dataset.
  config.dataset_name = 'custom_flexio'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = ['scenic.projects.pixel_llm.io.ops']

  tokenizer_path = common.BERT_TOKENIZER_PATH
  config.dataset_configs.tokenizer_weight_path = tokenizer_path  # Used in evaluation
  max_text_tokens = 40
  max_context_tokens = 8
  max_boxes = 100
  crop_size = 384

  # Train dataset(s).
  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.merge_sources = True
  config.dataset_configs.train.batch_before_merge = True
  config.dataset_configs.train.sources = [
      get_vg_train_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size),
  ]
  config.dataset_configs.train.postproc_spec = 'drop(["_seed"])'

  # Evaluation dataset(s).
  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.merge_sources = False
  config.dataset_configs.eval.sources = [
      get_vg_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size),
      get_vg_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='vg', split='test', task='loca'),
      get_coco_ref_eval_source(tokenizer_path, max_text_tokens, max_context_tokens, max_boxes, crop_size, dataset_name='refcocog_umd', split='validation'),
  ]
  config.dataset_configs.eval.postproc_spec = 'drop(["_seed"])'

  config.dataset_configs.densecap_configs = ml_collections.ConfigDict()
  config.dataset_configs.densecap_configs.test_annotation_path = common.VG_DENSECAP_TEST_ANNOTATION
  config.dataset_configs.loca_configs = ml_collections.ConfigDict()
  config.dataset_configs.loca_configs.vg_loca_test = ml_collections.ConfigDict()
  config.dataset_configs.loca_configs.vg_loca_test.merge_gt_boxes = False
  config.dataset_configs.loca_configs.refcocog_umd_loca_validation = ml_collections.ConfigDict()
  config.dataset_configs.loca_configs.refcocog_umd_loca_validation.merge_gt_boxes = False
  config.dataset_configs.extra_meta_data = {
      'input_shape': [-1, crop_size, crop_size, 3],
      'num_classes': -1,
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

  config.model.det_loss_weight = 1.0
  config.model.box_decoder_feature_key = 'git_visual_features+sam_visual_features'
  # CenterNet2 parameters
  config.model.box_decoder_name = 'centernet2_det_decoder'
  config.model.box_decoder_args = ml_collections.ConfigDict()
  config.model.box_decoder_args.num_classes = -1
  config.model.box_decoder_args.strides = (8, 16, 32, 64, 128)
  config.model.box_decoder_args.roi_num_classes = 1
  config.model.box_decoder_args.hm_weight = 0.5
  config.model.box_decoder_args.reg_weight = 1.0
  config.model.box_decoder_args.score_thresh = 0.0001
  config.model.box_decoder_args.pre_nms_topk_train = 2000
  config.model.box_decoder_args.post_nms_topk_train = 1000
  config.model.box_decoder_args.pre_nms_topk_test = 1000
  config.model.box_decoder_args.post_nms_topk_test = 256
  config.model.box_decoder_args.iou_thresh = 0.9
  config.model.box_decoder_args.roi_matching_threshold = (0.6,)
  config.model.box_decoder_args.roi_nms_threshold = 0.5
  config.model.box_decoder_args.roi_post_nms_num_detections = 100
  s = 2
  config.model.box_decoder_args.fpn_range = (
      (0, 80 / s), (64 / s, 160 / s), (128 / s, 320 / s),
      (256 / s, 640 / s), (512 / s, 100000 / s))
  config.model.box_decoder_args.match_gt_thresh = 0.6

  config.model.num_text_proposals = 64
  config.model.num_detections = config.model.box_decoder_args.roi_post_nms_num_detections

  # text
  config.model.decode_per_node_beam_size = 1
  config.model.max_caption_length = max_text_tokens

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
  config.model.decode_method = 'greedy'
  config.model.decode_beam_size = 1
  # config.model.decode_method = 'beam'
  # config.model.decode_beam_size = 4
  # config.model.decode_per_node_beam_size = 2
  config.model.mult_caption_score = False
  config.model.max_caption_length = max_text_tokens

  config.model.point_predictor_name = 'none'

  config.weights='/path/to/pixel_llm_bert_trace',
  config.skip_wrong_shape = False
  # config.eval_only = True

  # Training.
  config.batch_size = 64
  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.05
  config.optimizer.skip_scale_and_bias_regularization = True
  config.optimizer.layerwise_decay = 0.8
  config.optimizer.num_layers = config.model.git_backbone_args.depth
  config.optimizer.decay_layer_prefix = 'git_backbone/blocks.'
  config.optimizer.decay_stem_layers = ['patch_embed.proj', 'pos_embed']
  config.frozen_params = (
      ('.*sam_backbone.*', 'sam_backbone'),
      ('.*prompt_encoder.*', 'prompt_encoder'),
      # ('.*git_backbone.*', 'git_backbone'),
      )

  # learning rate and training schedule
  config.num_training_steps = 40000
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.warmup_steps = 1000
  config.lr_configs.base_learning_rate = 1e-4

  config.checkpoint_steps = 5000
  config.log_eval_steps = 5000

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 50  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


