# Scenic
<div style="text-align: left">
<img align="right" src="https://raw.githubusercontent.com/google-research/scenic/main/images/scenic_logo.png" width="200" alt="scenic logo"></img>
</div>

*Scenic* is a codebase with a focus on research around attention-based models
for computer vision. Scenic has been successfully used to develop
classification, segmentation, and detection models for multiple modalities
including images, video, audio, and multimodal combinations of them.

More precisely, *Scenic* is a (i) set of shared light-weight libraries solving
tasks commonly encountered tasks when training large-scale (i.e. multi-device,
multi-host) vision models; and (ii) several *projects* containing fully
fleshed out problem-specific training and evaluation loops using these
libraries.

Scenic is developed in [JAX](https://github.com/google/jax) and uses
[Flax](https://github.com/google/flax).

### Contents
* [What we offer](#what-we-offer)
* [SOTA models and baselines in Scenic](#sota-models-and-baselines-in-scenic)
* [Philosophy](#philosophy)
* [Getting started](#getting-started)
* [Scenic component design](#scenic-component-design)
* [Citing Scenic](#citing-scenic)

## What we offer
Among others *Scenic* provides

* Boilerplate code for launching experiments, summary writing, logging,
  profiling, etc;
* Optimized training and evaluation loops, losses, metrics, bi-partite matchers,
  etc;
* Input-pipelines for popular vision datasets;
* [Baseline models](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines#scenic-baseline-models),
including strong non-attentional baselines.


## SOTA models and baselines in *Scenic*
There are some SOTA models and baselines in Scenic which were either developed
using Scenic, or have been reimplemented in Scenic:

Projects that were developed in Scenic or used it for their experiments:

* [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
* [OmniNet: Omnidirectional Representations from Transformers](https://arxiv.org/abs/2103.01075)
* [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135)
* [TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?](https://arxiv.org/abs/2106.11297)
* [Exploring the Limits of Large Scale Pre-training](https://arxiv.org/abs/2110.02095)
* [The Efficiency Misnomer](https://arxiv.org/abs/2110.12894)
* [Discrete Representations Strengthen Vision Transformer Robustness](https://arxiv.org/abs/2111.10493)
* [Pyramid Adversarial Training Improves ViT Performance](https://arxiv.org/abs/2111.15121)
* [VUT: Versatile UI Transformer for Multi-Modal Multi-Task User Interface Modeling](https://arxiv.org/abs/2112.05692)
* [CLAY: Learning to Denoise Raw Mobile UI Layouts for Improving Datasets at Scale](https://arxiv.org/abs/2201.04100)
* [Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455)
* [Multiview Transformers for Video Recognition](https://arxiv.org/abs/2201.04288)
* [PolyViT: Co-training Vision Transformers on Images, Videos and Audio](https://arxiv.org/abs/2111.12993)
* [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)
* [Learning with Neighbor Consistency for Noisy Labels](https://arxiv.org/abs/2202.02200)
* [Token Turing Machines](https://arxiv.org/pdf/2211.09119.pdf)
* [Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning](https://arxiv.org/pdf/2302.14115.pdf)
* [AVATAR: Unconstrained Audiovisual Speech Recognition](https://arxiv.org/abs/2206.07684)
* [Adaptive Computation with Elastic Input Sequence](https://arxiv.org/abs/2301.13195)
* [Location-Aware Self-Supervised Transformers for Semantic Segmentation](https://arxiv.org/abs/2212.02400)
* [How can objects help action recognition?](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_How_Can_Objects_Help_Action_Recognition_CVPR_2023_paper.html)
* [Verbs in Action: Improving verb understanding in video-language models](https://arxiv.org/abs/2304.06708)
* [Unified Visual Relationship Detection with Vision and Language Models](https://arxiv.org/abs/2303.08998)
* [UnLoc: A Unified Framework for Video Localization Tasks](https://arxiv.org/abs/2308.11062)
* [REVEAL: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory](https://arxiv.org/abs/2212.05221)
* [Audiovisual Masked Autoencoders](https://arxiv.org/abs/2212.05922)
* [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707)
* [Pixel Aligned Language Models](https://arxiv.org/abs/2312.09237)
* [A Generative Approach for Wikipedia-Scale Visual Entity Recognition](https://arxiv.org/abs/2403.02041)
* [Streaming Dense Video Captioning](https://arxiv.org/abs/2404.01297)
* [Dense Video Object Captioning from Disjoint Supervision](https://arxiv.org/abs/2306.11729)

More information can be found in [projects](https://github.com/google-research/scenic/tree/main/scenic/projects#list-of-projects-hosted-in-scenic).

Baselines that were reproduced in Scenic:

* [(ViT) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* [(DETR) End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
* [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
* [(CLIP) Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
* [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688)
* [Universal Transformers](https://arxiv.org/abs/1807.03819)
* [PonderNet](https://arxiv.org/abs/2107.05407)
* [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
* [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
* [(CenterNet) Objects as Points](https://arxiv.org/abs/1904.07850)
* [(SAM) Segment Anything](https://arxiv.org/abs/2304.02643)


More information can be found in [baseline models](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines#scenic-baseline-models).

<a name="philosophy"></a>
## Philosophy
*Scenic* aims to facilitate rapid prototyping of large-scale vision models. To
keep the code simple to understand and extend we prefer *forking and
copy-pasting over adding complexity or increasing abstraction*. Only when
functionality proves to be widely useful across many models and tasks it may be
upstreamed to Scenic's shared libraries.


<a name="getting_start"></a>
## Getting started
* See `projects/baselines/README.md` for a walk-through baseline models and
  instructions on how to run the code.
* If you would like to contribute to *Scenic*, please check out the
  [Philisophy](#philosophy), [Code structure](#code_structure) and
  [Contributing](CONTRIBUTING.md) sections.
  Should your contribution be a part of the shared libraries, please send us a
  pull request!


### Quickstart
You will need Python 3.9 or later. Download the code from GitHub

```shell
$ git clone https://github.com/google-research/scenic.git
$ cd scenic
$ pip install .
```

and run training for ViT on ImageNet:

```shell
$ python scenic/main.py -- \
  --config=scenic/projects/baselines/configs/imagenet/imagenet_vit_config.py \
  --workdir=./
```

Note that for specific projects and baselines, you might need to install extra
packages that are mentioned in their `README.md` or `requirements.txt` files.

[Here](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/common_lib/colabs/scenic_playground.ipynb)
is also a minimal colab to train a simple feed-forward model using Scenic.

<a name="code_structure"></a>
## Scenic component design
Scenic is designed to propose different levels of abstraction, to support
hosting projects that only require changing hyper-parameters by defining config
files, to those that need customization on the input pipeline, model
architecture, losses and metrics, and the training loop. To make this happen,
the code in Scenic is organized as either _project-level_ code,
which refers to customized code for specific projects or baselines or
_library-level_ code, which refers to common functionalities and general
patterns that are adapted by the majority of projects. The project-level
code lives in the `projects` directory.

<div align="center">
<img src="https://raw.githubusercontent.com/google-research/scenic/main/images/scenic_design.jpg" width="900" alt="scenic design"></img>
</div>

### Library-level code
The goal is to keep the library-level code minimal and well-tested and to avoid
introducing extra abstractions to support minor use-cases. Shared libraries
provided by *Scenic* are split into:

*   `dataset_lib`: Implements IO pipelines for loading and pre-processing data
    for common Computer Vision tasks and benchmarks (see "Tasks and Datasets"
    section). All pipelines are designed to be scalable and support multi-host
    and multi-device setups, taking care dividing data among multiple hosts,
    incomplete batches, caching, pre-fetching, etc.
*   `model_lib` : Provides
    *   several abstract model interfaces (e.g. `ClassificationModel` or
        `SegmentationModel` in `model_lib.base_models`) with task-specific
        losses and metrics;
    *   neural network layers in `model_lib.layers`, focusing on efficient
        implementation of attention and transformer layers;
    *   accelerator-friendly implementations of bipartite matching
        algorithms in `model_lib.matchers`.
*   `train_lib`: Provides tools for constructing training loops and implements
    several optimized trainers (classification trainer and segmentation trainer)
    that can be forked for customization.
*   `common_lib`: General utilities, like logging and debugging modules,
    functionalities for processing raw data, etc.

### Project-level code
Scenic supports the development of customized solutions for customized tasks and
data via the concept of "project". There is no one-fits-all recipe for how much
code should be re-used by a project. Projects can consist of only configs and
use the common models, trainers, task/data that live in library-level code, or
they can simply fork any of the mentioned functionalities and redefine, layers,
losses, metrics, logging methods, tasks, architectures, as well as training and
evaluation loops. The modularity of library-level code makes it flexible for
projects to fall placed on any spot in the "run-as-is" to "fully customized"
spectrum.

Common baselines such as a ResNet and Vision Transformer (ViT) are implemented
in the [`projects/baselines`](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines)
project. Forking models in this directory is a good starting point for new
projects.


## Citing Scenic
If you use Scenic, you can cite our [white paper](https://openaccess.thecvf.com/content/CVPR2022/html/Dehghani_Scenic_A_JAX_Library_for_Computer_Vision_Research_and_Beyond_CVPR_2022_paper.html).
Here is an example BibTeX entry:

```bibtex
@InProceedings{dehghani2021scenic,
    author    = {Dehghani, Mostafa and Gritsenko, Alexey and Arnab, Anurag and Minderer, Matthias and Tay, Yi},
    title     = {Scenic: A JAX Library for Computer Vision Research and Beyond},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
    pages     = {21393-21398}
}
```

_Disclaimer: This is not an official Google product._
