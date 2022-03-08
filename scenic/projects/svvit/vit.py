"""ViT Classification model."""

from scenic.model_lib.base_models.classification_model import ClassificationModel

from scenic.projects.baselines import vit


class ViTClassificationModel(ClassificationModel):
  """ViT model for classification task."""

  def build_flax_model(self):
    return vit.ViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        dtype='float32',
    )
