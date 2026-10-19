# Scenic
<div style="text-align: left">
<img align="right" src="https://raw.githubusercontent.com/google-research/scenic/main/images/scenic_logo.png" width="200" alt="Scenic logo"></img>
</div>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

*Scenic* 是一个专注于计算机视觉领域基于注意力机制（Attention-based）模型研究的代码库。Scenic 已成功应用于开发涵盖图像、视频、音频以及多模态组合的分类、分割与检测模型。

更准确地说，*Scenic* 包含两个核心部分：(i) 一套共享的轻量级基础库，用于解决训练大规模（即多设备、多主机）视觉模型时常见的共性任务；以及 (ii) 多个独立的*项目（projects）*，其中包含使用这些基础库构建的、针对特定问题的完整训练与评估循环。

Scenic 基于 [JAX](https://github.com/jax-ml/jax) 开发，并使用 [Flax](https://github.com/google/flax) 构建神经网络。

### 目录 (Contents)

* [核心功能](#what-we-offer)
* [Scenic 中的 SOTA 模型与基线](#sota-models-and-baselines-in-scenic)
* [设计理念](#philosophy)
* [快速入门](#getting-started)
* [Scenic 组件架构设计](#scenic-component-design)
* [引用 Scenic](#citing-scenic)

<a id="what-we-offer"></a>
## 核心功能 (What we offer)
*Scenic* 主要提供以下核心能力：

* 用于启动实验、写入汇总指标（Summary Writing）、日志记录、性能分析（Profiling）等的标准化脚手架代码；
* 经过优化的训练与评估循环、损失函数、评估指标、二分图匹配器（Bi-partite Matchers）等；
* 面向主流计算机视觉数据集的高效数据输入流水线（Input-pipelines）；
* 丰富的[基线模型（Baseline models）](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines#scenic-baseline-models)，包括强有力的非注意力机制基线模型。

<a id="sota-models-and-baselines-in-scenic"></a>
## *Scenic* 中的 SOTA 模型与基线
Scenic 中包含了大量前沿 SOTA 模型与基线模型，它们要么直接基于 Scenic 研发，要么已在 Scenic 中完成了复现：

使用 Scenic 开发或开展实验的研究项目：

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

更多详情请参阅 [projects（项目列表）](https://github.com/google-research/scenic/tree/main/scenic/projects#list-of-projects-hosted-in-scenic)。

在 Scenic 中复现的经典基线模型：

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

更多详情请参阅 [baseline models（基线模型说明）](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines#scenic-baseline-models)。

<a name="philosophy"></a>
<a id="philosophy"></a>
## 设计理念 (Philosophy)
*Scenic* 旨在加速大规模视觉模型的快速原型设计。为了使代码保持简洁易懂且易于扩展，相比于*增加代码复杂度或引入过度抽象*，我们更倾向于*派生（Forking）与复制修改（Copy-pasting）*。只有当某项功能在众多模型和任务中被证明具有广泛的通用价值时，我们才会将其合入 Scenic 的共享基础库中。

<a name="getting_start"></a>
<a id="getting-started"></a>
## 快速入门 (Getting started)

* 请参阅 `projects/baselines/README.md`，了解基线模型的详细导览以及运行代码的具体步骤。
* 如果您希望向 *Scenic* 贡献代码，请先阅读[设计理念](#philosophy)、[代码结构](#code_structure)以及[贡献指南](CONTRIBUTING.md)章节。如果您的贡献属于共享基础库的一部分，欢迎向我们提交 Pull Request！

### 快速上手 (Quickstart)
您需要安装 Python 3.9 或更高版本。从 GitHub 下载代码并安装：

```shell
$ git clone https://github.com/google-research/scenic.git
$ cd scenic
$ pip install .
```

随后即可在 ImageNet 上运行 ViT 训练：

```shell
$ python scenic/main.py -- \
  --config=scenic/projects/baselines/configs/imagenet/imagenet_vit_config.py \
  --workdir=./
```

请注意，对于特定的研究项目和基线模型，您可能需要根据其各自目录下的 `README.md` 或 `requirements.txt` 文件安装额外的依赖包。

此外，[这里](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/common_lib/colabs/scenic_playground.ipynb)还提供了一个极简的 Colab 笔记本，演示如何使用 Scenic 训练一个简单的前馈网络模型。

<a name="code_structure"></a>
<a id="code_structure"></a>
<a id="scenic-component-design"></a>
## Scenic 组件架构设计 (Scenic component design)
Scenic 的设计初衷是提供不同层级的抽象：既能支持仅需通过定义配置文件来调整超参数的轻量级项目，也能支持需要对数据输入流水线、模型架构、损失函数、评估指标乃至整个训练循环进行深度定制的复杂研究项目。为了实现这一目标，Scenic 中的代码被组织为两个层级：_项目级（project-level）_代码（指针对特定研究项目或基线模型的定制化代码）与_基础库级（library-level）_代码（指被大多数项目广泛采用的通用功能与标准设计模式）。项目级代码均存放在 `projects` 目录下。

<div align="center">
<img src="https://raw.githubusercontent.com/google-research/scenic/main/images/scenic_design.jpg" width="900" alt="scenic design"></img>
</div>

### 基础库级代码 (Library-level code)
我们的目标是保持基础库级代码精简且经过充分测试，避免为了支持少数边缘用例而引入多余的抽象。*Scenic* 提供的共享基础库分为以下几个模块：

*   `dataset_lib`：实现用于常见计算机视觉任务和基准测试的数据加载与预处理 IO 流水线（详见“Tasks and Datasets”章节）。所有流水线均采用可扩展设计，原生支持多主机（multi-host）与多设备（multi-device）分布式环境，自动处理多主机间的数据切分、不完整批次（incomplete batches）、缓存与预取（pre-fetching）等细节。
*   `model_lib`：提供：
    *   位于 `model_lib.base_models` 中的多个抽象模型接口（例如 `ClassificationModel` 或 `SegmentationModel`），内置特定任务的损失函数与评估指标；
    *   位于 `model_lib.layers` 中的神经网络层，重点提供注意力机制与 Transformer 层的高效实现；
    *   位于 `model_lib.matchers` 中的面向硬件加速器优化的二分图匹配算法实现。
*   `train_lib`：提供用于构建训练循环的工具，并实现了多个经过深度优化的训练器（分类训练器与分割训练器），可直接派生（fork）以进行定制开发。
*   `common_lib`：通用工具库，如日志记录与调试模块、原始数据处理功能等。

### 项目级代码 (Project-level code)
Scenic 通过“项目（project）”的概念支持针对特定任务和数据集开发定制化解决方案。对于一个项目应当复用多少基础库代码，并没有一刀切的标准：项目可以仅包含配置文件，并直接调用基础库中的通用模型、训练器及任务/数据流水线；也可以直接派生（fork）上述任意模块，重新定义网络层、损失函数、评估指标、日志方式、任务定义、模型架构乃至训练与评估循环。基础库级代码的高度模块化设计，使得各个项目可以在“开箱即用（run-as-is）”到“完全深度定制（fully customized）”的光谱上自由选择最合适的位置。

诸如 ResNet 和 Vision Transformer (ViT) 等常见基线模型均在 [`projects/baselines`](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines) 项目中实现。从该目录派生模型是启动新研究项目的绝佳起点。

<a id="citing-scenic"></a>
## 引用 Scenic (Citing Scenic)
如果您在研究中使用了 Scenic，欢迎引用我们的[白皮书论文](https://openaccess.thecvf.com/content/CVPR2022/html/Dehghani_Scenic_A_JAX_Library_for_Computer_Vision_Research_and_Beyond_CVPR_2022_paper.html)。
以下为 BibTeX 引用示例：

```bibtex
@InProceedings{dehghani2021scenic,
    author    = {Dehghani, Mostafa and Gritsenko, Alexey and Arnab, Anurag and Minderer, Matthias and Tay, Yi},
    title     = {Scenic: A JAX Library for Computer Vision Research and Beyond},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
    pages     = {21393-21398}
}
```

_免责声明：这不是 Google 的官方产品。_

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月19日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
