## Scenic baseline models
This directory contains several baseline models implemented in Scenic.
These models span a range of architectures, tasks and modalities.

They include:

 * [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT) for image classification. [[Official Implementation](https://github.com/google-research/vision_transformer#vision-transformer)]
 * [Detection Transformer](https://arxiv.org/abs/2005.12872) (DETR) for object detection. [[Official PyTorch Implementation](https://github.com/facebookresearch/detr)]
 * [Deformable Detection Transformer](https://arxiv.org/abs/2010.04159) (Deformable DETR) for object detection. [[Official PyTorch Implementation](https://github.com/fundamentalvision/Deformable-DETR)]
 * [MLP-Mixer](https://arxiv.org/abs/2105.01601) an all-MLP model for image classification. [[Official Implementation](https://github.com/google-research/vision_transformer#mlp-mixer)]
 * [CLIP](https://arxiv.org/abs/2103.00020) for learning visual concepts from natural language supervision [[Official Implementation](https://github.com/openai/CLIP/tree/main/clip)]
 * [BERT](https://arxiv.org/abs/1810.04805) for language understanding. [[Official TF Implementation](https://github.com/google-research/bert)]
 * [Residual Networks](https://arxiv.org/abs/1512.03385) (ResNet) for image classification.
 * [Big Transfer ResNet](https://arxiv.org/abs/1912.11370) (BitResNet) for image classification. [[Official Implementation](https://github.com/google-research/big_transfer)]
 * [UNet](http://arxiv.org/abs/1505.04597) for semantic segmentation.
 * [Axial-ResNet](https://arxiv.org/abs/2003.07853) for image classification. [[Official TF Implementation](https://github.com/csrhddlam/axial-deeplab)]
 * [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688) for  shape classification, part segmentation and normal estimation tasks.
 * [Universal Transformers](https://arxiv.org/abs/1807.03819) for sequence modeling with adaptive computation.
 * [PonderNet](https://arxiv.org/abs/2107.05407) for sequence modeling with adaptive computation.
 * [CenterNet](https://arxiv.org/abs/1904.07850) and [CenterNet2](https://arxiv.org/abs/2103.07461) for object detection.
 * [SAM](https://arxiv.org/abs/2304.02643) for prompt-based segmentation

## Model Zoo

### Vision Transformers
We share checkpoints of models from the following papers, trained on various
datasets:

- **ViT**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **ViT-AugReg:** [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
  For the `ImageNet-21K`-pre-trained checkpoints we are using the "recommended
  checkpoints" (see paper section 4.5). The `ImageNet Accuracy` numbers are
  after fine-tuning (resolution 224px) as described in Appendix B. For more
  information see https://github.com/google-research/vision_transformer/

| Model | Dataset | Pretraining | ImageNet Accuracy | Checkpoint |
|-------|:-:|:-:|:-:|:-:|
| ViT-B/16            | ImageNet |       -        |  73.7* |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_16_ImageNet1k) |
| ViT-AugReg-B/16     | ImageNet |       -        |  79.7 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT-AugReg_B_16_ImageNet1k) |
| ViT-B/32            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_32_ImageNet21k) |
| ViT-B/16            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_16_ImageNet21k) |
| ViT-L/32            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_L_32_ImageNet21k) |
| ViT-L/16            |     -    |  ImageNet-21K  |   -   |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ViT_L_16_ImageNet21k) |
| ViT-AugReg-B/32     |     -    |  ImageNet-21K  |  79.1 |  [Link](https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz) |
| ViT-AugReg-B/16     |     -    |  ImageNet-21K  |  84.0 |  [Link](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz) |

*Note that 73.7 is the accuracy on ImageNet1k validation set, for a model that
is trained for 90 epochs from scratch. The scores reported in [How to train your ViT?](https://arxiv.org/abs/2106.10270) paper for vanilla ViT (74.1) is with 300
epochs of pre-training on ImageNet1k followed by fine-tuning on ImageNet1k.


The AugReg params can be directly loaded into a `train_state`:

```python
train_state_with_augreg_params = model.load_augreg_params(
    train_state,
    # Can read directly from storage bucket. Filename must start with model name
    # ("B_32-" in this case).
    'gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz',
    # Config is checked against AugReg config.
    config.model)
```


### ResNet
We share checkpoints of models from the following papers, trained on ImageNet:

- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **BiTResNet**: [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

| Model | Dataset | Pretraining | ImageNet Accuracy | Checkpoint |
|-------|:-:|:-:|:-:|:-:|
| ResNet50         | ImageNet |       -        |  76.1 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k) |
| BiTResNet50      | ImageNet |       -        |  77.0 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/BiTResNet50_ImageNet1k) |


### DETR
Please check [DETR directory](detr) for more information and link to download
pretrained checkpoints.

### Deformable DETR

Please check [Deformable DETR directory](deformable_detr) for more information
and link to download pretrained checkpoints.

### CLIP
Please check [CLIP directory](clip) for more information and link to download
pretrained checkpoints.


### BERT
Please check [BERT directory](bert) for more information and link to download
pretrained checkpoints.

### Universal Transformers
Please check [Universal Transformer directory](universal_transformer) for more
information.

### PonderNet
Please check [PonderNet directory](pondernet) for more information.

### CenterNet
Please check [CenterNet directory](centernet) for more information and link to download
pretrained checkpoints.

### SAM
Please check [SAM directory](segment_anything) for more information and link to
download pretrained checkpoints.
