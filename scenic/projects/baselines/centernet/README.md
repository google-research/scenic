CenterNet
==

This folder contains a Jax implementation of object detection using
CenterNet or CenterNet2.

Papers

 - CenterNet: https://arxiv.org/abs/1904.07850
 - CenterNet2: https://arxiv.org/abs/2103.07461

Pytorch code:

  - https://github.com/xingyizhou/CenterNet2


#### Install

In the Scenic root folder, run

```
pip install -r scenic/projects/baselines/centernet/requirements.txt
```

#### Dataset setup

Datasets are handled by [Tensorflow Dataset](https://www.tensorflow.org/datasets/catalog/overview) (tfds).
If the dataset used is already in tfds (e.g., COCO),
no additional steps are needed to setup the dataset.
The dataset will be downloaded automatically at the first run.
If the dataset used is not in tfds (e.g., Objects365), we'll need to convert the dataset format to
[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format first.
Todo this, we'll need to first convert the dataset to
[COCO format](https://cocodataset.org/#format-data), and run

```
python scenic/projects/baselines/centernet/tools/build_coco_tfrecord.py \
--input_json /path/to/instance/annotation.json \
--image_path /path/to/image/folder/ \
--output_path /path/to/output/instance.tfrecord
```

#### ImageNet pretrained checkpoint setup

For [ConvNeXt](https://arxiv.org/abs/2201.03545) backbones, run [this colab](notebooks/convert_convnext_weights.ipynb)
to convert ImageNet pretrained checkpoints to Jax.
For VitDet backbones, run [this colab](notebooks/convert_d2_vitdet_weights.ipynb)
to convert [MAE](https://arxiv.org/abs/2111.06377) pretrained checkpoints to Jax.
Update the `config.weights` in each config to the converted weights before running.

#### Training and evaluation

To train a model with a config, e.g., `centernet2_CXT_LSJ_4x`, run

```
python scenic/projects/baselines/centernet/main.py -- \
  --config=scenic/projects/baselines/centernet/configs/centernet2_CXT_LSJ_4x.py \
  --workdir=output/centernet2_CXT_LSJ_4x/
```
By default our model runs in multiple GPU/ TPU machines.
To run on fewer devices, reducing the batch size and learning rate accordingly
following linear learning rate rule is fine.
