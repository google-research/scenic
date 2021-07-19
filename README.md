## What is **Scenic**?

*Scenic* is a codebase with a focus on research around attention-based models
for computer vision. Scenic has been successfully used to develop
classificaiton, segmentation and detection models models for images, videos and
other modalities, including multimodal setups.

More precisely, *Scenic* is a (i) set of shared light-weight libraries solving
tasks commonly encountered tasks when training large-scale (i.e. multi-device,
multi-host) vision models; and (ii) a number of *projects* containing fully
fleshed out problem-specific training and evaluation loops using these
libraries.

Scenic is developed in JAX and uses [Flax](https://github.com/google-research/flax).

### What we offer
Among others *Scenic* provides
* Boilerplate code for launching experiments, summary writing, logging,
  profiling, etc;
* Optimized training and evaluation loops, losses, metrics, bi-partite matchers,
  etc;
* Input-pipelines for popular vision datasets;
* Baseline models, including strong non-attentional baselines.

Furthermore, a core of active users / Scenic owners
* Ensures that the code and (selected) models remain well-tested, correct and
  efficient. Scenic implementations are carefully chosen and optimised for TPUs.
* Helps setting up new projects.
* Is available to provide tips and pointers via code reviews.
* Offers an easy vehicle for open sourcing. *Scenic* is (almost) open source.
  This makes open sourcing projects within *Scenic* especially easy.
* Provides a path to deployment in production via JAX2TF.

## Philisophy
*Scenic* aims to facilitate rapid prototyping of large-scale vision models. To
keep the code simple to undrstand and extend we prefer *forking and copy-pasting
over adding complexity or increasing abstraction*. Only when functionality
proves to be widely useful across many models and tasks it may be upstreamed to
Scenic's shared libraries.

## Code structure
Shared libraries provided by *Scenic*  are split into:
* `dataset_lib` : Implements IO pipelines for loading and pre-processing data
  for common Computer Vision tasks and benchmarks (see "Tasks and Datasets"
  section). All pipelines are designed to be scalable and support multi-host and
  multi-device setups, taking care dividing data among multiple hosts,
  incomplete batches, caching, pre-fetching, etc.
* `model_lib` : Provides (i) several abstract model interfaces (e.g.
  `ClassificationModel` or `SegmentationModel` in `model_lib.base_models`) with
  task-specific losses and metrics; (ii) neural network layers in
  `model_lib.layers`, focusing on efficient implementation of attention and
  transfomer layers; and (iii) accelerator-friedly implementations of bipartite
  matching algorithms in `model_lib.matchers`.
* `train_lib` : Provides tools for constructing training loops and implements
  several example trainers (classification trainer and segmentation trainer).
* `common_lib` : Utilities that do not belong anywhere else.

### Projects
Models built on top of *Scenic* exist as separate projects. Model-specific code
such as configs, layers, losses, networks or training and evaluation loops exist
as a seprate project.

Common baselines such as a ResNet or a Visual Transformer (ViT) are implemented
in the `projects/baselines` project. Forking this directory is a good starting
point for new projects.

There is no one-fits-all recipe for how much code should be re-used by project.
Project can fall anywhere on the wide spectrum of code re-use: from defining new
configs for an existing model to redefining models, training loop, logging, etc.

## Getting started
* See `projects/baselines` for details on how to run a model.
* If you would like to to contribute to *Scenic*, please check out the
  "Philisopy" and "Code structure" sections. Should your contribution be a part
  of the shared libraries, come talk to us and / or send us a CL.
* Ready to start hosting your project within *Scenic*? Send us a CL creating the
  project directory. It should specify the OWNERS of the project and contain a
  README.
