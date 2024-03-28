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
r"""OWL v1 CLIP zero-shot text conditional detection training config.

Run training:
python -m scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=/tmp/training \
  --config=scenic/projects/owl_vit/configs/clip_l14_with_masks.py

"""
import ml_collections

CANONICAL_CHECKPOINT = 'gs://scenic-bucket/owl_vit/checkpoints/clip_vit_l14_with_masks_6c17944'

DETECTION_FEATURES = ('boxes', 'crowd', 'image', 'instance_labels',
                      'instance_text_labels', 'negative_labels',
                      'negative_text_labels', '_seed', 'seed')


def get_train_preproc_spec(
    *,
    input_size: int,
    num_instances: int,
    max_queries: int,
    max_query_length: int = 16,
    min_area_fraction: float = 0.0,
    iou_threshold: float = 0.9):
  """Constructs training preprocess string."""
  ops = (
      f'keep({DETECTION_FEATURES})'
      f'|random_flip_left_right'
      f'|random_crop(min_area_fraction={min_area_fraction})'
      f'|resize_with_pad(size={input_size})'
      f'|add_random_negative_labels(total_num_negatives=50)'
      f'|canonicalize_text_labels'
      f'|remove_forbidden_labels'
      f'|crop_or_pad({input_size}, {num_instances})'
      f'|crop_or_pad_meta_data({num_instances}, {num_instances})'
      f'|add_random_prompts'
      f'|remove_promptability_marker'
      f'|single_to_multi_label(max_num_labels={num_instances})'
      f'|merge_overlapping_instances(iou_threshold={iou_threshold})'
      f'|add_query_set(lower=True, max_queries={max_queries},'
      f' include_negatives=True)'
      f'|clip_tokenize_queries(max_token_len={max_query_length})')
  return ops


def get_eval_preproc_spec(
    *,
    input_size: int,
    num_instances: int,
    max_queries: int,
    max_query_length: int = 16,
    ):
  """Constructs training preprocess string."""
  return (
      f'resize_with_pad(size={input_size})'
      f'|canonicalize_text_labels'
      f'|crop_or_pad({input_size}, {num_instances})'
      f'|crop_or_pad_meta_data({num_instances}, {num_instances})'
      f'|single_to_multi_label(max_num_labels={num_instances})'
      f'|add_query_set(lower=True, max_queries={max_queries},'
      f' include_negatives=True)'
      f'|clip_tokenize_queries(max_token_len={max_query_length})')


def get_config(init_mode='train'):
  """Returns the configuration for text-query-based detection using OWL-ViT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'owl_vit_detection'

  # Dataset.
  config.dataset_name = 'owl_vit'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.input_size = 672
  config.dataset_configs.input_range = None
  config.dataset_configs.num_instances = 100
  config.dataset_configs.max_queries = 100
  config.dataset_configs.mask_size = 32
  config.dataset_configs.max_query_length = 16
  config.dataset_configs.min_area_fraction = 0.6
  config.dataset_configs.iou_threshold = 0.9
  config.dataset_configs.add_random_negatives = True
  config.dataset_configs.total_num_negatives = 50
  config.dataset_configs.prefetch_to_device = 2

  # For best performance, the shuffle buffer should be large, e.g. 10_000, but
  # this will require >50GB RAM.
  config.dataset_configs.shuffle_buffer_size = 1_000

  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.tfds_names = ['lvis']
  config.dataset_configs.train.splits = ['train']
  config.dataset_configs.train.dataset_probs = [1.0]
  config.dataset_configs.train.decoder_kwarg_list = ({},)
  config.dataset_configs.train.preproc_spec = get_train_preproc_spec(
      input_size=config.dataset_configs.input_size,
      num_instances=config.dataset_configs.num_instances,
      max_queries=config.dataset_configs.num_instances,
      max_query_length=config.dataset_configs.max_query_length,
      min_area_fraction=config.dataset_configs.min_area_fraction)
  # When using mosaics, use an input_size that is divisible by all mosaic_sizes.
  config.dataset_configs.train.mosaic_sizes = (1, 2, 3)
  config.dataset_configs.train.mosaic_probs = (.4, .3, .3)

  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.tfds_names = ['lvis']
  config.dataset_configs.eval.splits = ['validation']
  config.dataset_configs.eval.dataset_probs = [1.0]
  config.dataset_configs.eval.decoder_kwarg_list = ({},)
  config.dataset_configs.eval.preproc_spec = get_eval_preproc_spec(
      input_size=config.dataset_configs.input_size,
      num_instances=config.dataset_configs.num_instances,
      max_queries=config.dataset_configs.num_instances,
      max_query_length=config.dataset_configs.max_query_length)

  config.eval_top_k_preds = 300  # Only return the top-k predictions.
  config.data_dtype_str = 'float32'

  # Model.
  config.matcher = 'hungarian_cover_tpu'

  config.model = ml_collections.ConfigDict()
  config.model.normalize = True

  config.model.body = ml_collections.ConfigDict()
  config.model.body.type = 'clip'
  config.model.body.variant = 'vit_l14'
  config.model.body.merge_class_token = 'mul-ln'
  config.model.box_bias = 'both'

  # CLIP stochastic depth.
  config.model.body.text_stochastic_droplayer_rate = 0.1
  config.model.body.vision_stochastic_droplayer_rate = 0.2

  # Mask head.
  config.model.mask_head = ml_collections.ConfigDict()
  config.model.mask_head.mask_size = config.dataset_configs.get_ref('mask_size')
  config.model.mask_head.roi_align_num_parallel = 8
  config.model.mask_head.num_training_boxes = 64
  config.model.mask_head.stop_box_gradients = True
  config.model.mask_head.stop_image_gradients = True
  config.model.mask_head.num_mlp_layers_backbone_features = 1
  config.model.mask_head.image_resnet_width = 1.0
  config.model.mask_head.image_resnet_depth = (2, 2, 2, 2)
  config.model.mask_head.mask_resnet_width = 1.0
  config.model.mask_head.mask_resnet_depth = (2, 2, 2, 2)
  config.model.mask_head.add_image_coords = True
  config.model.mask_head.add_mask_coords = True
  config.model.mask_head.resnet_out_width_mult = 1
  config.model.mask_head.backbone_out_width_mult = 1

  # Loss.
  config.bbox_loss_coef = 1.0
  config.giou_loss_coef = 1.0
  config.class_loss_coef = 1.0
  config.focal_loss = True
  config.focal_gamma = 2.0
  config.focal_alpha = 0.3
  config.prior_prob = 0.01  # Prior prob of predicting not padding.
  config.normalization = 'per_example'  # 'per_example' or 'global'.

  # Training.
  config.trainer_name = 'text_zero_shot_detection'
  config.num_training_steps = 70_000
  config.batch_size = 256
  config.rng_seed = 0

  # Image backbone + head training configuration.
  sched = ml_collections.ConfigDict()
  sched.re = '(?!backbone/clip/text/.*)(.*)'  # Negative lookahead.
  sched.lr_configs = ml_collections.ConfigDict({  # Learning rate.
      'learning_rate_schedule': 'compound',
      'factors': 'constant*linear_warmup*cosine_decay',
      'steps_per_cycle': config.num_training_steps,
      'total_steps': config.num_training_steps,
      'warmup_steps': 1000,  # Necessary for higher LR and large batch size.
      'base_learning_rate': 2e-5,
  })

  # Text backbone training configuration.
  sched_txt = ml_collections.ConfigDict()
  sched_txt.re = '(backbone/clip/text/.*)'
  sched_txt.lr_configs = ml_collections.ConfigDict({
      'learning_rate_schedule': 'compound',
      'factors': 'constant*linear_warmup*cosine_decay',
      'steps_per_cycle': config.get_ref('num_training_steps'),
      'total_steps': config.get_ref('num_training_steps'),
      'warmup_steps': 1000,  # Necessary for higher LR and large batch size.
      'base_learning_rate': 2e-6,
  })

  # Configure both learning rate schedules.
  config.schedule = ml_collections.ConfigDict({
      'img_heads': sched,
      'txt': sched_txt,
  })

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scale_by_adam'
  optim.optax_configs = ml_collections.ConfigDict({  # Optimizer settings.
      'b1': 0.9,
      'b2': 0.999,
  })

  # Gradient clipping.
  optim.max_grad_norm = 1.0
  optim.per_example_clipping = True
  optim.optax_grad_pmean = True  # For per-example gradients Optax calls pmean.

  # Explicit WD (not via an optimizer).
  optim.weight_decay = 0.0
  optim.weight_decay_decouple = True

  config.optimizer = optim

  assert (optim.per_example_clipping or config.normalization != 'per_example'
          'Per example clipping only makes sense with local normalization')

  # Init.
  config.init_from = ml_collections.ConfigDict()
  if init_mode == 'train':
    config.init_from.codebase = 'clip'
  elif init_mode == 'canonical_checkpoint':
    config.init_from.checkpoint_path = CANONICAL_CHECKPOINT
  else:
    raise ValueError('Unknown init_mode: {}'.format(init_mode))

  # Logging.
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 100  # Train summary steps.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 2000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.log_eval_steps = 4000

  return config


