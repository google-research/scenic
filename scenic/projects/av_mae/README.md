### Audiovisual Masked Autoencoders

This repository is the JAX implementation of our ICCV 2023 paper,
[Audiovisual Masked Autoencoders](https://arxiv.org/abs/2212.05922).

Audiovisual Masked Autoencoders (AV-MAE) pretrains models on video and audio
data jointly, and shows improvements in both unimodal and multimodal downstream
tasks.

#### Getting Started

This project, like others in Scenic, uses [configuration files](configs).

To pretrain a model on AudioSet, run the following command:

```shell
$ python -m scenic.projects.av_mae.main \
  --config=scenic/projects/av_mae/configs/audioset/pretrain.py \
  --workdir=av_mae/
```

And then to finetune this model, run:

```shell
$ python -m scenic.projects.av_mae.main \
  --config=scenic/projects/av_mae/configs/audioset/finetune.py \
  --workdir=av_mae/
```

Make sure to set `config.init_from.checkpoint_path` to the pretrained model
when finetuning.

#### Model Zoo

The following table contains AV-MAE checkpoints trained on various datasets.
Checkpoints are provided as Scenic checkpoints compatible with
[Flax](https://github.com/google/flax).

| Dataset  | Model size | Pretraining modalities | Pretrained model  | Finetuning modalities | Finetuned model   | mAP / Accuracy |
|----------|------------|------------------------|-------------------|-----------------------|-------------------|----------------|
| AudioSet | Large      | audio, video           | [config](configs/audioset/pretrain.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual/checkpoint) | audio, video          | [config](configs/audioset/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_audiovisual/checkpoint) |  51.8              |
|          |            |                        |                   | audio                 | [config](configs/audioset/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_audio/checkpoint) |    46.6            |
|          |            |                        |                   | video                 | [config](configs/audioset/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_video/checkpoint) |  31.1             |
|          |            | audio                  | [config](configs/audioset/pretrain.py#L144) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audio/checkpoint) | audio                 | [config](configs/audioset/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audio_finetuned_audio/checkpoint) |  46.4              |
| VGGSound | Large      | audio, video           | [config](configs/vggsound/pretrain.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual/checkpoint) | audio, video          | [config](configs/vggsound/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_audiovisual/checkpoint) |  65.0              |
|          |            |                        |                   | audio                 | [config](configs/vggsound/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_audio/checkpoint) |    57.2            |
|          |            |                        |                   | video                 | [config](configs/vggsound/finetune.py) [checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_video/checkpoint) |  50.3             |

#### Reference

If you use this project, please cite the following BibTeX entry:

```
@inproceedings{georgescu2023audiovisual,
  title={Audiovisual Masked Autoencoders},
  author={Georgescu, Mariana-Iuliana and Fonseca, Eduardo and Ionescu, Radu Tudor and Lucic, Mario and Schmid, Cordelia and Arnab, Anurag},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
