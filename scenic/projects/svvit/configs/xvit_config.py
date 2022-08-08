# pylint: disable=line-too-long
r"""Default configs for X-ViT on structural variant classification using pileups.

"""

import ml_collections
from scenic.projects.svvit.google import dataset_meta_data

_TRAIN_SIZE = 30_000 * 18
VARIANT = 'Ti/4'

HIDDEN_SIZE = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'H': 1280}
NUM_HEADS = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}
MLP_DIM = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096, 'H': 5120}
NUM_LAYERS = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24, 'H': 24}


def get_config():
  """Returns the X-ViT experiment configuration for SV classification."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'sv-xvit'

  # Dataset.
  config.dataset_name = 'pileup_window'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.train_path = ''
  config.dataset_configs.eval_path = ''
  config.dataset_configs.test_path = ''

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'xvit_classification'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.hidden_size = HIDDEN_SIZE[version]
  config.model.patches.size = [int(patch), int(patch)]
  config.model.mlp_dim = MLP_DIM[version]
  config.model.num_layers = NUM_LAYERS[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.transformer_encoder_configs = ml_collections.ConfigDict()
  config.model.transformer_encoder_configs.type = 'global'
  config.model.attention_fn = 'performer'
  config.model.attention_configs = ml_collections.ConfigDict()
  config.model.attention_configs.attention_fn_cls = 'generalized'
  config.model.attention_configs.attention_fn_configs = None
  config.model.attention_configs.num_heads = NUM_HEADS[version]
  config.model.num_heads = NUM_HEADS[version]

  # Training.
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 20
  config.log_eval_steps = 100
  config.batch_size = 512
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # Learning rate.
  steps_per_epoch = _TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 3e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-6
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  # Evaluation:
  config.global_metrics = [
      'truvari_recall_events',
      'truvari_precision_events',
      'truvari_recall',
      'truvari_precision',
      'gt_concordance',
      'nonref_concordance',
  ]

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  # Dataset related hyper parameters.
  dataset_names = hyper.sweep('config.dataset_name', [
      'pileup_window',
  ])
  train_paths = hyper.sweep('config.dataset_configs.train_path', [
      dataset_meta_data.DATASET_PATHS['ref_right']['del']['single']
      ['hgsvc2_train'],
  ])
  eval_paths = hyper.sweep('config.dataset_configs.eval_path', [
      dataset_meta_data.DATASET_PATHS['ref_right']['del']['single']
      ['hgsvc2_test'],
  ])
  dataset_domains = hyper.zipit([dataset_names, train_paths, eval_paths])

  # Model related hyper parameters.
  hidden_size = hyper.sweep('config.model.hidden_size', [
      HIDDEN_SIZE['Ti'],
      HIDDEN_SIZE['S'],
      HIDDEN_SIZE['B'],
  ])
  num_heads = hyper.sweep('config.model.num_heads', [
      NUM_HEADS['Ti'],
      NUM_HEADS['S'],
      NUM_HEADS['B'],
  ])
  mlp_dim = hyper.sweep('config.model.mlp_dim', [
      MLP_DIM['Ti'],
      MLP_DIM['S'],
      MLP_DIM['B'],
  ])
  num_layers = hyper.sweep('config.model.num_layers', [
      NUM_LAYERS['Ti'],
      NUM_LAYERS['S'],
      NUM_LAYERS['B'],
  ])
  model_domains = hyper.zipit([hidden_size, num_heads, mlp_dim, num_layers])

  return hyper.product([model_domains, dataset_domains])
