"""Class File Registry for the AV-MAE project."""

from scenic.model_lib import models
from scenic.projects.av_mae import mbt
from scenic.projects.av_mae import vit
from scenic.projects.av_mae import vivit
from scenic.projects.av_mae import vivit_multimodal
from scenic.projects.baselines import vit as baseline_vit
from scenic.projects.vivit import model as baseline_vivit


def get_model_cls(model_name):
  """Returns the model class for training."""
  if model_name == 'vit_multilabel_classification':
    return baseline_vit.ViTMultiLabelClassificationModel
  elif model_name == 'vit_multilabel_classification_mae':
    return vit.ViTMAEMultilabelFinetuning
  elif model_name == 'vit_classification_mae':
    return vit.ViTMAEClassificationFinetuning
  elif model_name == 'vit_masked_autoencoder':
    return vit.ViTMaskedAutoencoderModel
  elif model_name == 'vivit_masked_autoencoder':
    return vivit.ViViTMaskedAutoencoderModel
  elif model_name == 'vivit_classification':
    return baseline_vivit.ViViTClassificationModel
  elif model_name == 'vivit_classification_mae':
    return vivit.ViViTMAEClassificationFinetuningModel
  elif model_name == 'vivit_multimodal_masked_autoencoder':
    return vivit_multimodal.ViViTMultiMaskedAutoencoderModel
  elif model_name == 'mbt_classification':
    return mbt.MBTClassificationModel
  elif model_name == 'mbt_multilabel_classification':
    return mbt.MBTMultilabelClassificationModel
  else:
    return models.get_model_cls(model_name)
