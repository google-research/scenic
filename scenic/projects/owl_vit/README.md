OWL-ViT: Open-World Object Detection with Vision Transformers
==
<img src="data/text_cond_wiki_stillife_1.gif" alt="OWL-ViT text inference demo" width="600"/>

<img src="data/owl_vit_schematic.png" alt="OWL-ViT model schematic" width="600"/>

OWL-ViT is an **open-vocabulary object detector**. Given an image and a free-text query, it finds objects matching that query in the image. It can also do **one-shot object detection**, i.e. detect objects based on a single example image. OWL-ViT reaches state-of-the-art performance on both tasks, e.g. **31% zero-shot LVIS APr** with a ViT-L/14 backbone.

[[Paper]](https://arxiv.org/abs/2205.06230)
[[Minimal Colab]](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb)
[[Playground Colab]](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_inference_playground.ipynb)

**Update (2022-10-14):** Added [training](#training) and [evaluation](#evaluation) code.
<br>
**Update (2022-07-06):** Extended TensorFlow-conversion [Colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_Export_JAX_model_to_TensorFlow_SavedModel.ipynb) with examples for conversion to TFLite.
<br>
**Update (2022-06-22):** Added [Playground Colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_inference_playground.ipynb) for interactive exploration of the model, including image-conditioned detection.
<br>
**Update (2022-05-31):** Added [Colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_Export_JAX_model_to_TensorFlow_SavedModel.ipynb) showing how to export models to TensorFlow.

## Contents
Below, we provide pretrained checkpoints, example Colabs, training code and evaluation code.

To get started, check out the [minimal example Colab notebook](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb), which shows all steps necessary for running inference, including installing Scenic, instantiating a model, loading a checkpoint, preprocessing input images, getting predictions, and visualizing them.

Table of contents:

* [Pretrained checkpoints](#pretrained-checkpoints)
* [Colabs](#colabs)
  * [Minimal example](#minimal-example)
  * [Inference playground](#inference-playground)
  * [Conversion to TensorFlow](#conversion-to-tensorflow)
* [Installation](#installation)
* [Training](#training)
* [Evaluation](#evaluation)
* [Reference](#reference)

## Pretrained checkpoints

OWL-ViT models and their pre-trained checkpoints are specified in [configuration files](https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/configs). Checkpoint files are compatible with [Flax](https://github.com/google/flax). We provide the following variants, both as JAX/Flax checkpoints and as `tf.SavedModel`s:

| Backbone | Pre-training | LVIS AP | LVIS APr | Config | Size | JAX Checkpoint | tf.SavedModel |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-B/32 | CLIP         | 19.3    | 16.9     | [clip_b32](https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/configs/clip_b32.py) | 583 MiB | [download](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc) | [gs](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc_tf_model) |
| ViT-B/16 | CLIP         | 20.8    | 17.1     | [clip_b16](https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/configs/clip_b16.py) | 581 MiB | [download](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_b16_6171dab) | [gs](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_b16_6171dab_tf_model) |
| ViT-L/14 | CLIP         | 34.6    | 31.2     | [clip_l14](https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/configs/clip_l14.py) | 1652 MiB | [download](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_l14_d83d374) | [gs](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_l14_d83d374_tf_model) |

## Colabs

### Minimal example
The [Minimal Example Colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb) shows all steps necessary for running inference, including installing Scenic, instantiating a model, loading a checkpoint, preprocessing input images, getting predictions, and visualizing them.

### Inference Playground
The [Playground Colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_inference_playground.ipynb) allows interactive exploration of the model. It supports both text-conditioned (open-vocabulary) and image-conditioned (one-shot) prediction:

<img src="data/text_cond_wiki_stillife_1.gif" alt="OWL-ViT text inference demo" height="200" style="margin:0px 30px"/>
<img src="data/image_cond_wiki_circuits_1.gif" alt="OWL-ViT image inference demo" height="200"/>

### Conversion to TensorFlow
OWL-ViT models can be converted to TensorFlow using the [`tf.saved_model`](https://www.tensorflow.org/guide/saved_model) API. The [Export Colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_Export_JAX_model_to_TensorFlow_SavedModel.ipynb) shows how to do this. For the public checkpoints, we provide `tf.SavedModel`s above (see [Pretrained checkpoints](#pretrained-checkpoints)).

## Installation

The code has been tested on Debian 4.19 and Python 3.7. For information on how to install JAX with GPU support, see [here](https://github.com/google/jax#installation).

```shell
git clone https://github.com/google-research/scenic.git
cd ~/scenic
python -m pip install -vq .
python -m pip install -r scenic/projects/owl_vit/requirements.txt

# For GPU support:
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Training

### Detection training
To train an OWL-ViT model with a CLIP-initialized backbone on detection, use:

```shell
python -m scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=/tmp/training \
  --config=scenic/projects/owl_vit/configs/clip_b32.py
```

Local TFDS data dirs can be specified like this:

```shell
python -m scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=/tmp/training \
  --config=scenic/projects/owl_vit/configs/clip_b32.py \
  --config.dataset_configs.train.decoder_kwarg_list='({"tfds_data_dir": "//your/data/dir"},)' \
  --config.dataset_configs.eval.decoder_kwarg_list='({"tfds_data_dir": "//your/data/dir"},)'
```

### Fine-tuning
To fine-tune a previously trained OWL-ViT model on your dataset of interest, use:

```shell
python -m scenic.projects.owl_vit.main \
  --alsologtostderr=true \
  --workdir=/tmp/training \
  --config=scenic/projects/owl_vit/configs/clip_b32_finetune.py
```

> NOTE: This config is just a starting point. Hyperparameters (especially learning rate and number of training steps, but also preprocessing, mosaics, and others) need to be tuned for each target dataset.

Adjust `config.dataset_configs.train.tfds_names` and related settings to your dataset of interest. You may have to write decoding ops similar to [`DecodeLvis`](https://github.com/google-research/scenic/blob/93fd069d969b3a3820b3b0b63f73fcff32dda093/scenic/projects/owl_vit/preprocessing/label_ops.py#L453) for your dataset. [`DecodeCocoExample`](https://github.com/google-research/scenic/blob/93fd069d969b3a3820b3b0b63f73fcff32dda093/scenic/projects/owl_vit/preprocessing/image_ops.py#L647) may be a good starting point. Make sure to handle negative examples correctly, e.g. by adding all classes that have no boxes in an image to the `MODALITIES.negative_text_labels` key of the feature dict for that image. (for non-federated datasets such as COCO).

## Evaluation
Since LVIS evaluation is slow, it is not included in the training loop. Model checkpoints can be evaluated as needed using a separate command.

For example, to evaluate the public B/32 checkpoint on LVIS, run:

```
python -m scenic.projects.owl_vit.evaluator \
  --alsologtostderr=true \
  --platform=gpu \
  --config=scenic/projects/owl_vit/configs/clip_b32.py \
  --checkpoint_path=gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc \
  --annotations_path=${HOME}/annotations/lvis_v1_val.json \
  --tfds_data_dir=//your/data/dir \
  --output_dir=/tmp/evaluator
```

## Reference

If you use OWL-ViT, please cite the [paper](https://arxiv.org/abs/2205.06230):

```
@article{minderer2022simple,
  title={Simple Open-Vocabulary Object Detection with Vision Transformers},
  author={Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby},
  journal={ECCV},
  year={2022},
}
```
