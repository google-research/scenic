CLAY: Learning to Denoise Raw Mobile UI Layouts for Improving Datasets at Scale
==

The CLAY pipeline is used to denoise raw mobile UI layouts by removing invalid
objects and assigning semantically meaningful types to the valid objects. This
repo hosts the codes for the Transformer-based object typing model. The CLAY
dataset used to train and evaluate the models can be downloaded
[here](https://github.com/google-research-datasets/clay).
Details can be found in the [paper](https://arxiv.org/abs/2201.04100).

## Getting Started

CLAY object typing model and training jobs are defined by [configuration files](configs).

To start, generate the training dataset according to [here](https://github.com/google-research/google-research/tree/master/clay).

An example command-line to train the object typing model using this [config file](configs/detr.py)
is

```
python scenic/projects/layout_denoise/main.py -- \
  --config=scenic/projects/layout_denoise/configs/detr.py \
  --workdir=clay_object_typing_model/
```

## Reference

If you use CLAY, please cite our paper:

```
@InProceedings{clay,
  title     = "Learning to Denoise Raw Mobile UI Layouts for Improving Datasets at Scale",
  booktitle = "Proceedings of the 2022 {CHI} Conference on Human Factors in
               Computing Systems",
  author    = "Li, Gang and Baechler, Gilles and Tragut, Manuel and Li, Yang",
  publisher = "Association for Computing Machinery",
  pages     = "1--13",
  month     =  may,
  year      =  2022,
  address   = "New Orleans, LA, USA",
  url = {https://doi.org/10.1145/3491102.3502042}
}
```
