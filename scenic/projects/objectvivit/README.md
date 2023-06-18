# How can objects help action recognition?

This project implements ObjectViViT, which uses object detection results from
off-the-shelf object detectors to help action recognition.

- First, we propose an object-guided token sampling strategy that enables us to drop certain input tokens with minimal impact on accuracy.
- Second, we propose an object-aware attention module that enriches the feature with object information to improve recognition accuracy.

> [**How can objects help action recognition?**](http://arxiv.org/abs/xxxx.xxxxx),
> Xingyi Zhou, Anurag Arnab, Chen Sun, Cordelia Schmid,
> *CVPR 2023*

## Getting Started

First install scenic following
[here](https://github.com/google-research/scenic#quickstart).
Then install additional dependency:

```
pip install -r scenic/projects/objectvivit/requirements.txt
```

Setup datasets following [DATA.md](DATA.md).

To train and evaluate a model, first download [VideoMAE](https://arxiv.org/abs/2203.12602)
pretrained checkpoints from
their [model zoo](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#something-something-v2),
and then convert them to JAX following `tools/convert_videomae_checkpoint.py`.
After replacing the checkpoint path in the config files, run

```
python -m scenic.projects.objectvivit.main \
  --config=scenic/projects/objectvivit/configs/ssv2_B16_object.py \
  --workdir=ssv2_B16_object/
```

We provide reference results on SSv2 dataset below:

|     config                    |  Accuracy |
|-------------------------------|-----------|
|ssv2_B16_baseline              | 66.1      |
|ssv2_B16_sampling (40% tokens) | 66.2      |
|ssv2_B16_object                | 67.4      |

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2023objects,
      title={How can objects help action recognition?},
      author={Zhou, Xingyi and Arnab, Anurag and Sun, Chen and Schmid, Cordelia},
      booktitle={CVPR},
      year={2023}
    }
