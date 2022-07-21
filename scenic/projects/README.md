## Contents
* [List of projects hosted in Scenic](#list-of-projects-hosted-in-scenic)
* [Scenic projects](#scenic-projects)


## List of projects hosted in Scenic

*   [ViViT](vivit)

    > ViViT is a family of pure-transformer based models for video
    > classification that achieved state-of-the-art results.
    > Details can be found in the [paper](https://arxiv.org/abs/2103.15691).

*   [TokenLearner](token_learner)

    > TokenLearner proposes dynamic tokenization of images and videos for faster
    > and more accurate video/image processing tasks. More can be found in
    > the [paper](https://arxiv.org/abs/2106.11297).


*   [FastViT](fast_vit)

    > FastViT is a project that aims at exploring ideas around making ViT faster
    > via using [efficient transformers](https://arxiv.org/abs/2009.06732), in
    > particular on higher resolution inputs (more tokens and thus longer
    > sequences).

*   [Omninet](omninet)

    > Omninet is a transformer model with
    > [omni-directional representations](https://arxiv.org/abs/2103.01075).

*   [CLAY](layout_denoise)

    > CLAY is a Transformer-based pipeline for mobile UI layout denoising. Read
    > more about this project in CLAY [paper](https://arxiv.org/abs/2201.04100).


*   [MBT](mbt)

    > MBT presents a transformer based architecture that uses "fusion
    > bottlenecks" for modality fusion at multiple layers.
    > Details can be found in the [paper](https://arxiv.org/abs/2201.04100).

*   [MTV](mtv)

    > MTV presents a state-of-the-art transformer based architecture for video
    > classification. MTV consists of separate encoders to represent different
    > views of the input video with lateral connections and a global encoder to
    > fuse information across views. More details are in the
    > [paper](https://arxiv.org/abs/2201.04288).

*   [OWL-ViT](owl_vit)

    > OWL-ViT is an open-vocabulary object detector that given an image and a
    > free-text query, it finds objects matching that query in the image. It can
    > also do one-shot object detection, i.e. detect objects based on a single
    > example image. More details are in the
    > [paper](https://arxiv.org/abs/2205.06230).

*   [NCR](ncr)

    > NCR is a regularization method which encourages the network to make
    > similar predictions for similar vectors in the feature space.
    > Details can be found in the [paper](https://arxiv.org/abs/2202.02200),
    > where we used this method to learn with noisy labels.

*   [PCT](pointcloud)

    > Point Cloud Transformer (PCT) is a Transformer-based model for
    > performing inference (classification/segmentation) for point cloud data.
    > Details can be found in the [paper](https://arxiv.org/abs/2012.09688).


<a name="projects"></a>
## Scenic projects
A typical project consists of models, trainers, configs, a runner, and some
utility functions developed for the project.

### Models
Models are entities that define the network architecture, loss function, and
metrics. Network architectures are built using Flax `nn.Modules`. Common loss
functions and metrics can be included via a
[Base Model](../model_lib/README.md#base_model), or within the project
itself for more specific use-cases.

To be accessible by the trainer, a model newly-defined by a project needs to be
registered *within a specific project*. As an exception, the baseline models
are registered directly in `model_lib.models`.

### Trainers
Trainers implement the training and evaluation loops of the model. There are
already standard trainers that are provided in Scenic for classification,
segmentation, and adaptation (located in the `train_lib` module).
These trainers are directly registered  in `train_lib_deprecated/trainers` and
given the careful optimization of these trainers for fast and efficient training
on accelerators (in particular TPUs), they can be forked by projects for further
customization. Projects need to register the new trainers they define within
their project, or they can simply use the standard Scenic trainers when no
modification is needed.

### Configs
Config files are used to configure experiments. They define (hyper-)parameters
for the selected model, trainer, and dataset (e.g. number of layers, frequency
of logging, etc).

### Binaries
Binaries bind models, trainers, and datasets together based on the config and
start the training. Usually, this is a `main.py` within the project that also
contains the registry for the project specific models and trainers. Note that
baselines make use of Scenic's default binary `main.py`.

### Registries
There are three types of objects that can be registered in Scenic:
`model`, `trainer`, and `dataset`. A registry could be any simple data structure
that maps a string name to an object, for instance, a python dictionary.

Scenic defines a dataset registry that uses ad-hoc importing to lazy-load
the code for the input pipeline of a requested dataset. This registry lives in
`dataset_lib/datasets.py`. There are common trainers and models that are
registered in  `train_lib_deprecated/trainers.py` and `model_lib/models.py`. However,
a project can define its own dataset, model, and trainer and make a small
registry for these objects within the project, e.g. in the project's `main.py`
so that the right model, trainer, and dataset are selectable using the
configs specified in the config file.
