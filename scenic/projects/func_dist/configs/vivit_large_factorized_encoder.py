r"""Config for training ViViT on regression.

"""


from absl import logging
import ml_collections

NUM_CLASSES = 174
SSV2_TRAIN_SIZE = 68913
SSV2_VAL_SIZE = 24777


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'vivit_regression_large_fe_ssv2'

  # Dataset.
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'
  config.dataset_name = 'ssv2_regression_tfrecord'
  config.dataset_configs.base_dir = 'path/to/dataset/root'
  config.dataset_configs.tables = {
      'train': 'something-something-v2-train.rgb.tfrecord@128',
      'validation': 'something-something-v2-validation.rgb.tfrecord@128',
      'test': 'something-something-v2-validation.rgb.tfrecord@128',
  }
  config.dataset_configs.examples_per_subset = {
      'train': SSV2_TRAIN_SIZE,
      'validation': SSV2_VAL_SIZE,
      'test': SSV2_VAL_SIZE,
  }
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.included_tasks_path = None
  config.dataset_configs.train_metadata_path = None
  config.dataset_configs.validation_metadata_path = None

  # This is going to sample 3 consecutive frames and a future goal frame.
  # TODO(minttu): Add a config field for adding goal frame.
  config.dataset_configs.num_frames = 4
  config.dataset_configs.stride = 1
  config.dataset_configs.min_resize = 288
  config.dataset_configs.crop_size = 256
  config.dataset_configs.zero_centering = True

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.log_test_epochs = 5
  # The effective batch size per host when testing is num_test_clips * test_batch_size  # pylint: disable=line-too-long
  config.dataset_configs.num_test_clips = 1
  config.dataset_configs.test_batch_size = 8  # Needs to be num_local_devices
  config.multicrop_clips_per_device = 2
  # Leaving this empty means that a full test is done each time.
  # config.steps_per_test = 1000  # Number of test steps taken by each host.

  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = False
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1

  config.dataset_configs.augmentation_params.augment_goals = True

  # This does Mixup in the data-loader. Done on Numpy CPU, so its slow
  # config.dataset_configs.augmentation_params.do_mixup = False
  # config.dataset_configs.augmentation_params.mixup_alpha = 0.0

  # This does Mixup in the train loop. This is fast. But make sure that device
  # batch size is more than 1. On a 4x4 TPU, this means that your batch size
  # needs to be at least 64.
  # For Kinetics, we have not been using Mixup
  # config.mixup = ml_collections.ConfigDict()
  # config.mixup.alpha = 0.3

  config.dataset_configs.augmentation_params.do_rand_augment = True
  config.dataset_configs.augmentation_params.rand_augment_num_layers = 2
  config.dataset_configs.augmentation_params.rand_augment_magnitude = 20

  config.dataset_configs.prefetch_to_device = 2

  # Model: ViT-base
  config.model_name = 'vivit_regression'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = 1024
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]

  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'factorized_encoder'

  # Only for the Factorisation-v1 model.
  config.model.spatial_transformer = ml_collections.ConfigDict()
  config.model.spatial_transformer.num_heads = 16
  config.model.spatial_transformer.mlp_dim = 4096
  config.model.spatial_transformer.num_layers = 24
  config.model.temporal_transformer = ml_collections.ConfigDict()
  config.model.temporal_transformer.num_heads = 16
  config.model.temporal_transformer.mlp_dim = 4096
  config.model.temporal_transformer.num_layers = 4
  # End only for Factorisation-v1 model.

  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.stochastic_droplayer_rate = 0.3
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.patches.size = [16, 16, 1]
  config.model.temporal_encoding_config.kernel_init_method = 'reduce_sum_initializer'

  # Training.
  config.trainer_name = 'vivit_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.label_smoothing = None  # 0.3
  config.num_training_epochs = 35
  config.batch_size = 64
  config.rng_seed = 0

  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_config = None
  # The following is ViViT trained on full SSv2.
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.checkpoint_path = 'path/to/init/ckpt'
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_temporal_embedding_for_goal = False
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'resize'

  # Learning rate.
  steps_per_epoch = SSV2_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = steps_per_epoch // 2
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.5

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.checkpoint_steps = 500  # Checkpoint more frequently than a val epoch
  config.log_summary_steps = 100

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""

  batch_size = 64
  steps_per_epoch = SSV2_TRAIN_SIZE // batch_size
  default_warmup = steps_per_epoch // 2

  logging.info('Train set size: %d, Number of steps per epoch: %d',
               SSV2_TRAIN_SIZE, steps_per_epoch)

  common_params = hyper.product([
      hyper.zipit([
          hyper.sweep('config.trainer_name', ['vivit_trainer']),
          hyper.sweep('config.model_name', ['vivit_regression']),
      ]),
      hyper.sweep('config.optimizer', ['momentum_hp']),
      hyper.sweep('config.batch_size', [batch_size]),
      hyper.sweep('config.lr_configs.base_learning_rate', [0.1, 0.2]),
      # NB. If you change the number of epochs, then have to change the
      # number of steps per cycle in the cosine decay as well.
      hyper.zipit([
          hyper.sweep(
              'config.num_training_epochs', [20, 30]),
          hyper.sweep(
              'config.lr_configs.steps_per_cycle',
              [i * steps_per_epoch for i in [20, 30]]),
      ]),
      hyper.sweep('config.init_from.restore_positional_embedding', [True]),
      hyper.sweep('config.init_from.restore_temporal_embedding_for_goal',
                  [False]),
      #
      # Number of frames
      #
      hyper.sweep('config.dataset_configs.num_frames', [4, 11]),
      hyper.sweep('config.dataset_configs.stride', [1]),
      #
      hyper.sweep('config.lr_configs.warmup_steps', [5 * default_warmup]),
      hyper.sweep('config.dataset_configs.log_test_epochs', [2]),
      hyper.sweep('config.multicrop_clips_per_device', [2]),
      # Model params
      hyper.sweep('config.model.temporal_transformer.num_layers', [4]),
      # Init from K400 pre-trained model, and also Im21-K init model.
      hyper.zipit([
          hyper.sweep('config.init_from.checkpoint_format',
                      ['scenic']),
      ]),
      hyper.sweep('config.init_from.positional_embed_size_change', ['resize']),
  ])

  params = hyper.product([
      common_params,
      hyper.sweep('config.model.temporal_encoding_config.method', ['3d_conv']),
      hyper.sweep('config.init_from.restore_input_embedding', [True]),
      hyper.sweep(
          'config.model.patches.size',
          [
              [16, 16, 1],  # Effectively 16 input frames
          ]),
      hyper.sweep('config.model.temporal_encoding_config.kernel_init_method',
                  ['reduce_sum_initializer']),
  ])

  return params
