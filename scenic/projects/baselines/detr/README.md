## DEtection TRansformer (DETR)
This directory contains the implementation of DETR for [end-to-end object detection with transformers](https://arxiv.org/abs/2005.12872).
The code here uses JAX and Flax and follows the [official implementation of DETR in PyTorch](https://github.com/facebookresearch/detr).

### Additional Requirements:
The following command will install the required packages for DETR.
```shell
$ pip install -r scenic/projects/baselines/detr/requirements.txt
```

### Training DETR
In order to train DETR on COCO object detection, you can use the
`detr_config.py`  in the [configs directory](configs):

```shell
$ python scenic/projects/baselines/detr/main.py -- \
  --config=scenic/projects/baselines/detr/configs/detr_config.py \
  --workdir=./
```

In the config, you have to set the path to the pre-trained ResNet50 backbone
that you can download from [here](https://storage.googleapis.com/scenic-bucket/baselines/ResNet50_ImageNet1k).
(More information on other potential pre-trained backbones can be found [here](../baselines#resnet).)


### Checkpoint
We also share checkpoint of a DETR model trained on COCO dataset:

| Model | Task | Dataset | Average Precision | Checkpoint |
|-------|:-:|:-:|:-:|:-:|
| DETR | Object Detection | COCO | 0.4038 |  [Link](https://storage.googleapis.com/scenic-bucket/baselines/DETR_COCO_detection) |


### Alternative matcher
DETR uses bipartite matching loss to find a bipartite matching
between ground truth and prediction. This is required for end-to-end training to
enforce permutation-invariance, and guarantee that each target element
has a unique match. By default, the Hungarian algorithm is used for the matching,
however, there are alternative algorithms, like Sinkhorn that are more
accelerator friendly. We have used [OTT](https://github.com/google-research/ott)
and added [a config file with Sinkhorn matching](configs/detr_sinkhorn_config.py)
that achieves similar performance, with relatively higher speed than Hungarian.


### Acknowledgment
We would like to thank Dirk Weissenborn, Aravindh Mahendran, Sunayana Rane,
Rianne van den Berg, Olivier Teboul, Marco Cuturi for their amazing
help on implementing DETR in Scenic.
