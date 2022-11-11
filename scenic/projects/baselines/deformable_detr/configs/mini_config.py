r"""Mini config for COCO detection using DeformableDETR.
"""

import ml_collections


def get_config():
  """Returns the configuration for COCO using a mini DeformableDETR."""
  config = ml_collections.ConfigDict()

  # Dataset.
  config.dataset_name = 'coco_detr_detection'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.max_boxes = 10
  config.data_dtype_str = 'float32'

  # Model.
  config.model_dtype_str = 'float32'
  config.model_name = 'deformable_detr'
  config.matcher = 'hungarian'
  config.num_classes = 91
  config.embed_dim = 32
  config.enc_embed_dim = 32
  config.num_queries = 12
  config.num_feature_levels = 4
  config.num_heads = 4
  config.num_encoder_layers = 2
  config.num_decoder_layers = 2
  config.transformer_ffn_dim = 256
  config.num_enc_points = 1
  config.num_dec_points = 2
  config.backbone_num_filters = 16
  config.backbone_num_layers = 18
  config.dropout_rate = 0.1

  # Loss.
  config.aux_loss = True
  config.bbox_loss_coef = 1.0
  config.giou_loss_coef = 1.0
  config.class_loss_coef = 1.0
  config.focal_loss_alpha = 0.25
  config.focal_loss_gamma = 2.0

  # Training.
  config.rng_seed = 0

  # Optimization.
  config.optimizer_config = ml_collections.ConfigDict()
  config.optimizer_config.weight_decay = 1e-4
  config.optimizer_config.beta1 = 0.9
  config.optimizer_config.beta2 = 0.999
  config.optimizer_config.base_learning_rate = 2e-4
  config.optimizer_config.max_grad_norm = 0.1
  config.optimizer_config.learning_rate_decay_rate = 0.1
  config.optimizer_config.learning_rate_reduction = 0.1
  config.optimizer_config.learning_rate_decay_event = 1

  # Pretrained_backbone.
  config.load_pretrained_backbone = False
  config.freeze_backbone_batch_stats = False

  # Logging.
  config.write_summary = False  # don't write summary
  config.checkpoint = False  # don't do checkpointing
  config.checkpoint_steps = None
  config.debug_train = False  # don't debug  during training
  config.debug_eval = False  # don't debug during eval

  config.num_training_steps = 2
  config.log_eval_steps = 2
  config.steps_per_eval = 2
  config.num_training_epochs = None

  return config
