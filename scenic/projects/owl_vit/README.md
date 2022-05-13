OWL-ViT: Open-World Object Detection with Vision Transformers
==
<img src="data/owl_vit_schematic.png" alt="ViT+ model schematic" width="800"/>

OWL-ViT is an **open-vocabulary object detector**. Given an image and a free-text query, it finds objects matching that query in the image. It can also do **one-shot object detection**, i.e. detect objects based on a single example image. ViT+ reaches state-of-the-art performance on both tasks, e.g. **31% zero-shot LVIS APr** with a ViT-L/14 backbone.

[[Paper]](https://arxiv.org/abs/2205.06230) [[Colab]](notebooks/OWL_ViT_minimal_example.ipynb)

## Getting Started
We currently provide code for running inference with pre-trained checkpoints. Training code will follow soon.

The [minimal example Colab notebook](notebooks/OWL_ViT_minimal_example.ipynb) shows all steps necessary for running inference, including installing Scenic, instantiating a model, loading a checkpoint, preprocessing input images, getting predictions, and visualizing them.

## Model Variants

OWL-ViT models and their pre-trained checkpoints are specified in [configuration files](configs). Checkpoint files are compatible with [Flax](https://github.com/google/flax). We provide the following variants:

| Backbone | Pre-training | LVIS AP | LVIS APr | Config | Size | Checkpoint |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-B/32 | CLIP         | 19.3    | 16.9     | [clip_b32](configs/clip_b32.py) | 583 MiB | [download](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc) |
| ViT-B/16 | CLIP         | 20.8    | 17.1     | [clip_b16](configs/clip_b16.py) | 581 MiB | [download](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_b16_6171dab) |
| ViT-L/14 | CLIP         | 34.6    | 31.2     | [clip_l14](configs/clip_l14.py) | 1652 MiB | [download](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/clip_vit_l14_d83d374) |

## Reference

If you use OWL-ViT, please cite the [paper](https://arxiv.org/abs/2205.06230):

```
@article{minderer2022simple,
  title={Simple Open-Vocabulary Object Detection with Vision Transformers},
  author={Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby},
  journal={arXiv preprint arXiv:2205.06230},
  year={2022},
}
```
