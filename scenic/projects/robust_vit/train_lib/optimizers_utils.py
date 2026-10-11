"""Define our own optimizer for robust vit optimization.

Our optimizer only optimizes ViT and ignores the vqGAN part.
"""

from flax import optim as optimizers


def get_partial_optimizer(config) -> optimizers.OptimizerDef:
  """Return a optimizer class for optimization."""
  if 'adam' in config.optimizer:
    vit_optim = optimizers.Adam(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
    )
    vit_param = optimizers.ModelParamTraversal(lambda path, _: 'vit' in path)
  elif 'momentum_hp_vitonly' == config.optimizer:
    vit_optim = optimizers.Momentum(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta=config.optimizer_configs.get('momentum', 0.9))
    vit_param = optimizers.ModelParamTraversal(lambda path, _: 'vit' in path)

  if config.get('bert_train_mask', True):
    mask_codebook_param = optimizers.ModelParamTraversal(
        lambda path, _: 'mask_embedding' in path)
    #  fix first layer? or 'lp_embedding' in path
    mask_codebook_optim = optimizers.Adam(
        learning_rate=(config.lr_configs['base_learning_rate']),
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
    )
    opt_def = optimizers.MultiOptimizer((vit_param, vit_optim),
                                        (mask_codebook_param,
                                         mask_codebook_optim))

  elif config.get('finetune_embed_code', False):
    codebook_delta_param = optimizers.ModelParamTraversal(
        lambda path, _: 'code_embedding_delta' in path)
    codebook_2_optim = optimizers.Adam(
        learning_rate=(config.lr_configs['base_learning_rate'] /
                       config.get('finetune_embed_code_lr_decrease', 1)),
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
    )
    opt_def = optimizers.MultiOptimizer((vit_param, vit_optim),
                                        (codebook_delta_param,
                                         codebook_2_optim))
  elif config.get('retrain_embed_code', False):
    codebook_2_param = optimizers.ModelParamTraversal(
        lambda path, _: 'code_embedding_retrain' in path)
    codebook_2_optim = optimizers.Adam(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
    )
    opt_def = optimizers.MultiOptimizer((vit_param, vit_optim),
                                        (codebook_2_param, codebook_2_optim))
  elif config.get('finetune_gan', False):
    vqencoder_param = optimizers.ModelParamTraversal(
        lambda path, _: 'vqgan' in path)
    vqencoder_optim = optimizers.Adam(
        learning_rate=(config.lr_configs['base_learning_rate'] /
                       config.get('lr_decrease', 1)),
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
    )
    opt_def = optimizers.MultiOptimizer((vit_param, vit_optim),
                                        (vqencoder_param,
                                         vqencoder_optim))
  else:
    opt_def = optimizers.MultiOptimizer((vit_param, vit_optim))
  return opt_def
