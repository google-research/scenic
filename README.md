# Scenic

*Scenic* is a codebase with a focus on research around attention-based models
for computer vision. Scenic has been successfully used to develop
classification, segmentation, and detection models for multiple modalities
including images, video, audio, and multimodal combinations of them.

More precisely, *Scenic* is a (i) set of shared light-weight libraries solving
tasks commonly encountered tasks when training large-scale (i.e. multi-device,
multi-host) vision models; and (ii) a number of *projects* containing fully
fleshed out problem-specific training and evaluation loops using these
libraries.

Scenic is developed in [JAX](https://github.com/google/jax) and uses
[Flax](https://github.com/google/flax).

## What we offer
Among others *Scenic* provides

* Boilerplate code for launching experiments, summary writing, logging,
  profiling, etc;
* Optimized training and evaluation loops, losses, metrics, bi-partite matchers,
  etc;
* Input-pipelines for popular vision datasets;
* Baseline models, including strong non-attentional baselines.


## Papers using *Scenic*
Scenic can be used to reproduce the results from the following papers, which
were either developed using Scenic, or have been reimplemented in Scenic:

* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)


<a name="philosophy"></a>
## Philosophy
*Scenic* aims to facilitate rapid prototyping of large-scale vision models. To
keep the code simple to understand and extend we prefer *forking and
copy-pasting over adding complexity or increasing abstraction*. Only when
functionality proves to be widely useful across many models and tasks it may be
upstreamed to Scenic's shared libraries.


<a name="code_structure"></a>
## Code structure
Shared libraries provided by *Scenic*  are split into:

* `dataset_lib`: Implements IO pipelines for loading and pre-processing data
  for common Computer Vision tasks and benchmarks. All pipelines are designed to
  be scalable and support multi-host and multi-device setups, taking care of
  dividing data among multiple hosts, incomplete batches, caching, pre-fetching,
  etc.
* `model_lib`: Provides (i) several abstract model interfaces (e.g.
  `ClassificationModel` or `SegmentationModel` in `model_lib.base_models`) with
  task-specific losses and metrics; (ii) neural network layers in
  `model_lib.layers`, focusing on efficient implementation of attention and
  transfomer layers; and (iii) accelerator-friedly implementations of bipartite
  matching algorithms in `model_lib.matchers`.
* `train_lib`: Provides tools for constructing training loops and implements
  several example trainers (classification trainer and segmentation trainer).
* `common_lib`: Utilities that do not belong anywhere else.


### Projects
Models built on top of *Scenic* exist as separate projects. Model-specific code
such as configs, layers, losses, network architectures, or training and
evaluation loops exist as separate projects.

Common baselines such as a ResNet or a Visual Transformer (ViT) are implemented
in the `projects/baselines` project. Forking this directory is a good starting
point for new projects.

There is no one-fits-all recipe for how much code should be re-used by projects.
Projects can fall anywhere on the wide spectrum of code re-use: from defining
new configs for an existing model to redefining models, training loop, logging,
etc.


## Getting started
* See `projects/baselines/README.md` for a walk-through baseline models and
  instructions on how to run the code.
* If you would like to to contribute to *Scenic*, please check out the
  [Philisophy](#philosophy), [Code structure](#code_structure) and
  [Contributing](CONTRIBUTING.md) sections.
  Should your contribution be a part of the shared libraries, please send us a
  pull request!


### Quick start
Download the code from GitHub

```
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
```

and run training for ViT on ImageNet:

```
python scenic/main.py -- \
  --config=scenic/projects/baselines/configs/imagenet/imagenet_vit_config.py \
  --workdir=./
```

_Disclaimer: This is not an official Google product._
