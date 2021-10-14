## Scenic baseline models
This directory contains several baseline vision models implemented in Scenic.
These models span a range of architectures, as well as tasks.

They include:

 * Vision Transformer (ViT; https://arxiv.org/abs/2010.11929) for image
  classification.
 * MLP-Mixer (https://arxiv.org/abs/2105.01601) an all-MLP model for image
  classification.
 * Residual Networks (ResNet; https://arxiv.org/abs/1512.03385) and Big Transfer
  ResNet (BitResNet; https://arxiv.org/abs/1912.11370) for image classification.
 * UNet (http://arxiv.org/abs/1505.04597) for semantic segmentation.
 * Axial-ResNet (https://arxiv.org/abs/2003.07853) for image classification.


## Model Zoo

### Vision Transformers
We share checkpoints of models from the following papers, trained on various
datasets:

- **ViT**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **ViT-AugReg:** [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)

| Model | Dataset | Pretraining | ImageNet Accuracy | Checkpoint |
|-------|:-:|:-:|:-:|:-:|
| ViT-B/16            | ImageNet |       -        |  74.1 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_16_ImageNet1k) |
| ViT-AugReg-B/16     | ImageNet |       -        |  79.7 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT-AugReg_B_16_ImageNet1k) |
| ViT-B/32            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_32_ImageNet21k) |
| ViT-B/16            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_16_ImageNet21k) |
| ViT-L/32            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_L_32_ImageNet21k) |
| ViT-L/16            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_L_16_ImageNet21k) |


### ResNet
We share checkpoints of models from the following papers, trained on ImageNet:

- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **BiTResNet**: [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

| Model | Dataset | Pretraining | ImageNet Accuracy | Checkpoint |
|-------|:-:|:-:|:-:|:-:|
| ResNet50         | ImageNet |       -        |  76.1 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k) |
| BiTResNet50      | ImageNet |       -        |  77.0 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/BiTResNet50_ImageNet1k) |
