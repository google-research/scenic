# Discrete ViT

*Discrete_vit* is the codebase for experiments on improving the Robustness of ViT
model with integer input token representation,
currently supporting content robustness. The content robustness is described in paper
[Discrete Representations Strengthens Vision Transformer Robustness](https://arxiv.org/abs/2111.10493).

Robust_vit is developed in [JAX](https://github.com/google/jax) and uses
[Flax](https://github.com/google/flax) and Scenic code framework.


## Code structure

* `main`: The main function to call for running our code, which specifies the
dataset, model, trainer algorithm

* `model`: Our model specification

* `trainer`: Our training algorithm

* `robustness_evaluator`: defines the evaluation step and evaluator, which
will be run at evaluation time to replace training trainer.


## Getting Started

Discrete ViT models and training jobs are defined by [configuration files](configs).


An example command-line to train Discrete Only ViT-B/16
using this [config file](configs/robustness/imagenet_vqvit_reg_config.py)
is

```
python scenic/projects/robust_vit/main.py -- \
  --config=scenic/projects/robust_vit/configs/robustness/imagenet_vqvit_reg_config.py \
  --workdir=discrete_only_vit/
```

An example command-line to train Discrete+Continuous Token ViT-B/16
using this [config file](configs/robustness/imagenet_fuvqvit_reg_config.py)
is

```
python scenic/projects/robust_vit/main.py -- \
  --config=scenic/projects/robust_vit/configs/robustness/imagenet_fuvqvit_reg_config.py \
  --workdir=discrete_only_vit/
```

## Model Zoo

The following table contains pretrained Discrete ViT models trained on various datasets.
Checkpoints are provided as Scenic checkpoints compatible with
[Flax](https://github.com/google/flax), and also as
[Tensorflow SavedModels](https://www.tensorflow.org/guide/saved_model)

Please download the ImageNet pretrained VQGAN model [here](),
and ImageNet21K pretrained VQGAN model [here]().

| Model | Training Set | Config | Checkpoint |
|-------|--------------|--------|------------|
|Discrete Only ViT-B-16| ImageNet| [config file](configs/robustness/imagenet_vqvit_reg_config.py) | [checkpoint]() |
|Ours ViT-B-16| ImageNet| [config file](configs/robustness/imagenet_fuvqvit_reg_config.py) | [checkpoint]() |
|Discrete Only ViT-B-16| ImageNet21K| [config file](configs/robustness/imagenet21k_vqvit_augreg_config.py) [Finetune config file](configs/robustness/imagenet_vqvit_ft21k_reg_config.py) | [checkpoint]() |
|Ours ViT-B-16| ImageNet21K| [config file](configs/robustness/imagenet21k_fuvqvit_augreg_config.py) [Finetune config file](configs/robustness/imagenet_fuvqvit_ft21k_config.py) | [checkpoint]() |

## Reference
If you use Discrete ViT, please use the following BibTeX entry.

```
@article{mao2021discrete,
  title={Discrete Representations Strengthen Vision Transformer Robustness},
  author={Mao, Chengzhi and Jiang, Lu and Dehghani, Mostafa and Vondrick, Carl and Sukthankar, Rahul and Essa, Irfan},
  journal={arXiv preprint arXiv:2111.10493},
  year={2021}
}
```
