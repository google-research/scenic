# pylint: disable=line-too-long
r"""Default configs for BERT pretraining.

"""
# pylint: enable=line-too-long

import ml_collections
from scenic.projects.baselines.bert.configs.glue import glue_fewshot

VARIANT = 'BERT-B'

EMBEDDING_WIDTH = {'Ti': 128, 'S': 128, 'B': 768, 'L': 1024}
HIDDEN_SIZE = {'Ti': 128, 'S': 256, 'B': 768, 'L': 1024}
NUM_HEADS = {'Ti': 2, 'S': 4, 'B': 12, 'L': 16}
MLP_DIM = {'Ti': 512, 'S': 1024, 'B': 3072, 'L': 4096}
NUM_LAYERS = {'Ti': 6, 'S': 12, 'B': 12, 'L': 24}


def get_config():
  """Returns configuration for BERT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'bert'

  # Dataset.
  config.dataset_name = 'bert_wikibooks'
  config.data_dtype_str = 'float32'
  config.batch_size = 512
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  # Training data:
  config.dataset_configs.train_data_loader = ml_collections.ConfigDict()
  wikibooks_train_loader = config.dataset_configs.train_data_loader
  wikibooks_train_loader.seq_length = 512
  wikibooks_train_loader.max_predictions_per_seq = 76
  wikibooks_train_loader.use_next_sentence_label = True
  wikibooks_train_loader.use_position_id = False
  wikibooks_train_loader.use_v2_feature_names = False
  # Add path to training files containing tf.records.
  wikibooks_train_loader.input_path = ''
  wikibooks_train_loader.drop_remainder = True
  wikibooks_train_loader.shuffle_buffer_size = 100
  wikibooks_train_loader.cycle_length = None
  wikibooks_train_loader.block_length = 1
  wikibooks_train_loader.deterministic = None
  wikibooks_train_loader.sharding = True
  wikibooks_train_loader.enable_tf_data_service = False
  wikibooks_train_loader.tf_data_service_address = None
  wikibooks_train_loader.tfds_name = None
  wikibooks_train_loader.tfds_split = None
  wikibooks_train_loader.tfds_as_supervised = False
  wikibooks_train_loader.tfds_skip_decoding_feature = ''
  wikibooks_train_loader.global_batch_size = config.batch_size
  wikibooks_train_loader.prefetch_buffer_size = None  # Autotune.
  # Validation data:
  config.dataset_configs.val_data_loader = ml_collections.ConfigDict()
  wikibooks_val_loader = config.dataset_configs.val_data_loader
  wikibooks_val_loader.seq_length = 512
  wikibooks_val_loader.max_predictions_per_seq = 76
  wikibooks_val_loader.use_next_sentence_label = True
  wikibooks_val_loader.use_position_id = False
  wikibooks_val_loader.use_v2_feature_names = False
  # Add path to validation files containing tf.records.
  wikibooks_val_loader.input_path = ''
  wikibooks_val_loader.drop_remainder = False
  wikibooks_val_loader.cycle_length = None
  wikibooks_val_loader.block_length = 1
  wikibooks_val_loader.deterministic = None
  wikibooks_val_loader.sharding = True
  wikibooks_val_loader.enable_tf_data_service = False
  wikibooks_val_loader.tf_data_service_address = None
  wikibooks_val_loader.tfds_name = None
  wikibooks_val_loader.tfds_split = None
  wikibooks_val_loader.tfds_as_supervised = False
  wikibooks_val_loader.tfds_skip_decoding_feature = ''
  wikibooks_val_loader.global_batch_size = config.batch_size
  wikibooks_val_loader.prefetch_buffer_size = None  # Autotune.

  # Model.
  _, model_size = VARIANT.split('-')
  config.model_name = 'bert'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.stem = ml_collections.ConfigDict()
  config.model.stem.hidden_size = HIDDEN_SIZE[model_size]
  config.model.stem.embedding_width = EMBEDDING_WIDTH[model_size]
  config.model.stem.max_position_embeddings = 512
  config.model.stem.dropout_rate = 0.1
  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.num_heads = NUM_HEADS[model_size]
  config.model.encoder.mlp_dim = MLP_DIM[model_size]
  config.model.encoder.num_layers = NUM_LAYERS[model_size]
  config.model.encoder.attention_dropout_rate = 0.1
  config.model.encoder.dropout_rate = 0.1
  config.model.encoder.pre_norm = True
  config.model.head = ml_collections.ConfigDict()
  config.model.head.type = 'pretraining'
  config.model.head.hidden_size = HIDDEN_SIZE[model_size]

  # Training.
  config.trainer_name = 'bert_trainer'
  config.optimizer = 'adam'  # Change to adamw, when it's available in Scenic.
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.max_grad_norm = 1.0
  config.num_training_epochs = None
  config.num_training_steps = 1000_000
  config.log_eval_steps = 1000
  config.steps_per_eval = 64
  config.rng_seed = 42

  # Fewshot.
  config.fewshot = glue_fewshot.get_config(config.batch_size)
  config.fewshot.log_eval_steps = 50_000

  # Learning Rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * linear_warmup * linear_decay'
  config.lr_configs.warmup_steps = 10000
  config.lr_configs.total_steps = config.num_training_steps
  config.lr_configs.base_learning_rate = 1e-4

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 20000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  return config


