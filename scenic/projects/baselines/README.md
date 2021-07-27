## Scenic baseline models
This directory contains several baseline vision models implemented in Scenic.
These models span a range of architectures, as well as tasks.

They include:

 * Vision Transformer (ViT; https://arxiv.org/abs/2010.11929) for image
  classification (ImageNet-1k, ImageNet-21k), task adaptation and fewshot
  learning.
 * Residual Networks (ResNet; https://arxiv.org/abs/1512.03385) and Big Transfer
  ResNet (BitResNet; https://arxiv.org/abs/1912.11370) for image classification
  (ImageNet-1k).
 * UNet (http://arxiv.org/abs/1505.04597) for semantic segmentation (CityScapes).


## Setup
A typical project consists of models, trainers, configs and a runner.

### Models
Models (i.e. their architectures as Flax nn.Modules and corresponding subclasses
of the standard models in `model_lib.base_models`) are implemented in the
corresponding modules under `projects.baselines`.

To be accessible by the trainer models need to be registered *within a specific
project*. As an exception, the baseline models are registered directly in
`model_lib.models`.

### Trainers
Trainers implement the training and evaluation loops of the model. Baselines
provided with Scenic include classification, segmentation and adaptation
trainers (located in the `train_lib` module). As an exception, baseline trainers
are registered directly in `train_lib.trainers`.

### Configs
Config files are used to configure experiments. They define (hyper-)parameters
for the selected model, trainer and dataset (e.g. number of layers, frequency of
logging, etc).

### Binary
Binaries bind models, trainers and datasets together based on the config and
start the training. Baselines make use of Scenic's default binary `main.py`.

## Getting started
The best way to get started is to train a model yourself!

Here's how you can train a ViT model on ImageNet:

```
python main.py -- \
  --config=projects/baselines/configs/imagenet/imagenet_vit_config.py \
  --workdir=./
```

