## Audiovisual Masked Autoencoders

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
[Flax](https://github.com/google/flax), and in
Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format
for easy inference.

| Dataset  | Model size | Pretraining modalities | Pretrained model  | Finetuning modalities | Finetuned model   | mAP / Accuracy |
|----------|------------|------------------------|-------------------|-----------------------|-------------------|----------------|
| AudioSet | Large      | audio, video           | [Config](configs/audioset/pretrain.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual/tf_saved_model.zip) | audio, video          | [Config](configs/audioset/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_audiovisual/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_audiovisual/tf_saved_model.zip) |  51.8              |
|          |            |                        |                   | audio                 | [Config](configs/audioset/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_audio/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_audio/tf_saved_model.zip) |    46.6            |
|          |            |                        |                   | video                 | [Config](configs/audioset/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_video/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audiovisual_finetuned_video/tf_saved_model.zip) |  31.1             |
|          |            | audio                  | [Config](configs/audioset/pretrain.py#L144) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audio/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audio/tf_saved_model.zip) | audio                 | [Config](configs/audioset/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audio_finetuned_audio/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/audioset/as2m_selfsup_audio_finetuned_audio/tf_saved_model.zip) |  46.4              |
| VGGSound | Large      | audio, video           | [Config](configs/vggsound/pretrain.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual/tf_saved_model.zip) | audio, video          | [Config](configs/vggsound/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_audiovisual/checkpoint)  <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_audiovisual/tf_saved_model.zip) |  65.0              |
|          |            |                        |                   | audio                 | [Config](configs/vggsound/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_audio/checkpoint) <br /> [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_audio/tf_saved_model.zip) |    57.2            |
|          |            |                        |                   | video                 | [Config](configs/vggsound/finetune.py) <br /> [Checkpoint](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_video/checkpoint) <br />  [TF SavedModel](https://storage.googleapis.com/scenic-bucket/av_mae/vggsound/vggsound_selfsup_audiovisual_finetuned_video/tf_saved_model.zip) |  50.3             |

#### Using Tensorflow SavedModels

###### Pretrained models

Here, the inputs are audio waveforms (16kHz) and/or rgb frames, and the outputs
are token embeddings from the encoder of the model.

The model is called as follows:

```python
restored_tf_model = tf.saved_model.load(model_dir)
tf_output = restored_tf_model(tf_input)
tf_output_spec = tf_output['spectrogram']  # shape is [batch, num_tokens=496, hidden_dimension=1024].
tf_output_rgb = tf_output['rgb']  # shape is [batch, num_tokens=1568, hidden_dimension=1024].
```

where `tf_input = {'rgb': TensorSpec(shape=(None, 16, 224, 224, 3), dtype=tf.float32), 'waveform': TensorSpec(shape=(None, 160 000, 1), dtype=tf.float32)}`
for an input clip of 10s (as used for AudioSet).
Models pretrained on VGGSound use 8s inputs instead (128 000 samples).
Log-mel spectrograms are computed within the model.
For the model pretrained only with audio, the input signature is the same, but
only the `'waveform'` key is used.
A `None` shape means that any positive value can be used in the batch dimension.

And `tf_output['spectrogram']` has shape `(batch, 496, 1024)` for 10s inputs, or `(batch, 400, 1024)` for 8s input, where 496=62x8=TxF and 400=50x8=TxF squared 16x16-patches that fit in the incoming spectrogram
(T and F denote the number of time- and frequency bins in the spectrogram respectively).
Similarly, `tf_output['rgb']` usually has shape `(batch, 1568, 1024)`,
where 1568=14x14x8=HxWxD 16x16x2-patches that fit in the incoming 16 RGB frames.


###### Finetuned models

Here, the inputs are audio waveforms (16kHz) and/or rgb frames, and the outputs
are classification logits from the model.

The model is called as follows:

```python
restored_tf_model = tf.saved_model.load(model_dir)
tf_output = restored_tf_model(tf_input)  # shape is [batch, num_classes].
```

where `tf_input = {'rgb': TensorSpec(shape=(None, 32, 224, 224, 3), dtype=tf.float32), 'waveform': TensorSpec(shape=(None, 160 000, 1), dtype=tf.float32)}`
for an input clip of 10s (as used for AudioSet).
Models finetuned on VGGSound use 8s inputs instead (128 000 samples).
Log-mel spectrograms are computed within the model.
For the models finetuned with only one modality,
the input signature is the same, but only one key is used
(`'rgb'` or `'waveform'`).
A `None` shape means that any positive value can be used in the batch dimension.

`tf_output` has shape `(batch, num_classes)`, where the last dimension corresponds
to the classification logits (527 for AudioSet and 309 for VGGSound).


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
