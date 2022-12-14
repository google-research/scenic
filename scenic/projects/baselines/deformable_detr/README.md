## Deformable DEtection TRansformer (Deformable DETR)
This directory contains the implementation of Deformable DETR for [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).
The code here uses JAX and Flax and follows the [official implementation of Deformable DETR in PyTorch](https://github.com/fundamentalvision/Deformable-DETR). Note that we implement the iterative bounding box refinement but not the two-stage paradigm.

### Additional Requirements:
The following command will install the required packages for DETR.

```shell
$ pip install -r scenic/projects/baselines/deformable_detr/requirements.txt
```

### Training Deformable DETR
In order to train DETR on COCO object detection, you can use `coco_config.py`
(to run locally) or `xc_coco_config.py` (to run on Google Cloud) in the
[configs directory](configs). For example:

```shell
$ python scenic/projects/baselines/deformable_detr/main.py -- \
  --config=scenic/projects/baselines/deformable_detr/configs/coco_config.py \
  --workdir=./
```

In the config, you have to set the path to a pre-trained ResNet50 backbone.
You can download one from [here](https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k).
(More information on other potential pre-trained backbones can be found [here](../baselines#resnet).)


### Results
| Average Precision | Notes |
|:-----------------:|-------|
| 0.459 | The model is trained from PyTorch ResNet50_Weights.IMAGENET1K_V1. |
| 0.459 | The model is evaluated with the corresponding official PyTorch weights. |
