TokenLearner
==
![TokenLearner](data/tokenlearner.gif)

TokenLearner is a learable module to be placed within Transformer archituctures
for images and videos. Once placed, it significantly reduces the number of
tokens for all subsequent layers, thereby reducing the overall computation.
It simultaneously increases accuracy of the models by making the tokens dynamic
and adaptive to the input. It supports both image and video representation
models, such as [ViT](https://arxiv.org/pdf/2010.11929.pdf) and
[ViViT](https://arxiv.org/pdf/2103.15691.pdf).

TokenLearner achieved state-of-the-art results on four public datasets,
including Kinetics-400, Kinetics-600, Charades, and AViD. Details can be found
in the [paper](https://arxiv.org/abs/2106.11297).

## Getting Started

TokenLearner models and training jobs are defined by [configuration files](configs).

Alternatively, you can plug in the TokenLearnerModule (or TokenLearnerModuleV11)
from the [model file](model.py) into any of your Transformer architectures, and
benefit from it. Learning 8 or 16 tokens in the middle of the network is often
sufficient to maintain the accuracy while cutting the computation by half.

An example command-line to train a base ViT model on ImageNet (following the
settings in the original [ViT paper](https://arxiv.org/pdf/2010.11929.pdf))
using TokenLearner is:
```
python scenic/projects/token_learner/main.py -- \
  --config=scenic/projects/token_learner/configs/im1k_token_learner_config.py \
  --workdir=token_learner/
```

## Reference

If you use TokenLearner, please use the following BibTeX entry.

```
@InProceedings{ryoo2021tokenlearner,
  title={TokenLearner: Adaptive Space-Time Tokenization for Videos},
  author={Ryoo, Michael S. and Piergiovanni, AJ and Arnab, Anurag and Dehghani, Mostafa and Angelova, Anelia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
