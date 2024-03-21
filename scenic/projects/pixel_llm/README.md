Pixel Aligned Language Models
==
**PixelLLM**: Pixel-Aligned Language Model (PixelLLM) equips large language models with localization capability. The model is pre-trained on [Localized Narratives](https://google.github.io/localized-narratives/), to learn the alignment between
words and image pixels. PixelLLM can be applied to various localization tasks, for example, location-conditioned captioning when taking
location as input, and referring localization when generating locations as outputs.

This directory is the official implementation of PixelLLM introduced in the paper:
[**Pixel Aligned Language Models**](https://arxiv.org/abs/2312.09237)

[*Jiarui Xu*](https://jerryxu.net),
[*Xingyi Zhou*](https://xingyizhou.github.io/),
[*Shen Yan*](https://shenyann.github.io/),
[*Xiuye Gu*](https://laoreja.github.io/),
[*Anurag Arnab*](https://anuragarnab.github.io/),
[*Chen Sun*](https://chensun.me/index.html),
[*Xiaolong Wang*](https://xiaolonw.github.io/),
[*Cordelia Schmid*](https://scholar.google.com/citations?user=IvqCXP4AAAAJ&hl=en)

<img src="https://jerryxu.net/PixelLLM/figs/github/teaser.gif" alt="PixelLLM teaser" width="100%"/>

## Visual Results
<img src="https://jerryxu.net/PixelLLM/figs/github/trace.gif" alt="PixelLLM trace" width="100%"/>

## Links
* [Project Page](https://jerryxu.net/PixelLLM/) (with additional visual results)
* [arXiv Page](https://arxiv.org/abs/2312.09237)

## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@inproceedings{xu2023pixel,
  title={Pixel aligned language models},
  author={Xu, Jiarui and Zhou, Xingyi and Yan, Shen and Gu, Xiuye and Arnab, Anurag and Sun, Chen and Wang, Xiaolong and Schmid, Cordelia},
  booktitle={CVPR},
  year={2023}
}
```

## Model Zoo

<img src="https://jerryxu.net/PixelLLM/figs/github/ref.gif" alt="PixelLLM ref" width="49%"/>
### Referring Expression Localization and Segmentation

| Language Model                                | RefCOCO val |           | RefCOCO testA |           | RefCOCO testB |           | RefCOCO+ val |           | RefCOCO+ testA |           | RefCOCO+ testB |           | RefCOCOg val |           | RefCOCOg test |           | download       |
|-----------------------------------------------|-------------|-----------|---------------|-----------|---------------|-----------|--------------|-----------|----------------|-----------|----------------|-----------|--------------|-----------|---------------|-----------|----------------|
|                                               | box P@0.5   | mask cIoU | box P@0.5     | mask cIoU | box P@0.5     | mask cIoU | box P@0.5    | mask cIoU | box P@0.5      | mask cIoU | box P@0.5      | mask cIoU | box P@0.5    | mask cIoU | box P@0.5     | mask cIoU |                |
| [BERT](configs/bert/pixel_llm_bert_refseg.py) | 89.6        | 76.4      | 91.4          | 77.8      | 86.9          | 74.0      | 82.2         | 67.8      | 86.9           | 71.8      | 76.4           | 62.1      | 84.2         | 69.4      | 85.0          | 70.5      | [checkpoint]() |
| [T5-XL](configs/t5/pixel_llm_t5_refseg.py)    | 90.3        | 77.2      | 92.3          | 79.2      | 86.8          | 74.0      | 83.7         | 69.2      | 87.2           | 73.0      | 78.5           | 64.4      | 84.5         | 70.1      | 85.7          | 72.0      | [checkpoint]() |

<img src="https://jerryxu.net/PixelLLM/figs/github/densecap.gif" alt="PixelLLM densecap" width="49%"/>
### Dense Object Captioning and Location-conditioned Object Captioning

| Language Model                                  | Visual Genome |        |       | RefCOCOg |       | download       |
|-------------------------------------------------|---------------|--------|-------|----------|-------|----------------|
|                                                 | mAP           | METEOR | CIDEr | METEOR   | CIDEr |                |
| [BERT](configs/bert/pixel_llm_bert_densecap.py) | 17.4          | 20.0   | 148.0 | 14.8     | 86.6  | [checkpoint]() |
| [T5-XL](configs/t5/pixel_llm_t5_densecap.py)    | 17.5          | 20.1   | 149.0 | 15.3     | 92.0  | [checkpoint]() |

## Environment Setup

In the Scenic root folder, run

```
pip install -r scenic/projects/pixel_llm/requirements.txt
```

<!-- TODO(zhouxy): DenseVoc evaluator -->
For evaluation, you need to download captioning metrics files from [this repository](https://github.com/antoyang/captioning-metrics) and put them in the `metrics` folder. Note you will also need to download JAVA and specify the location to your Jre java bin in the [main](main.py) file.

Like other projects in Scenic, all model parameters, training sets and datasets are specified using [configuration files](configs).

To train a model with T5-XL text model, please download a pretrained T5-XL model from [T5X](https://github.com/google-research/t5x) and specify its path in [Scenic T5](https://github.com/google-research/scenic/tree/main/scenic/projects/t5).

To train model with BERT text model, please download BERT vocabulary from [Huggingface](https://huggingface.co/google-bert/bert-base-uncased/blob/main/vocab.txt) and specific its path in [scenic/projectsj/pixel_llm/configs/common.py](configs/common.py).

## Training

An example command-line to train PixelLLM on LN with [config file](configs/bert/pixel_llm_bert_trace.py) is

```shell
$ python -m scenic.projects.pixel_llm.main \
  --config=scenic/projects/vid2seq/configs/bert/pixel_llm_bert_trace.py \
  --workdir=pixel_llm_bert_trace/
```

To evaluate the model only, you need to add `config.eval_only=True` in the config file.

## Dataset setup

NOTE: Please update the dataset path in `scenic/projectsj/pixel_llm/configs/common.py` after finishing TFRecord preparation.

### Prepare Localized Narratives (LN) TFRecords

You need to download [COCO Images](https://cocodataset.org/#download) and [LN Annotations](https://google.github.io/localized-narratives/)

```shell
python scenic/projects/pixel_llm/tools/build_ln_tfrecord.py \
--output_dir ~/Datasets/LN \
--ln_anno_path ~/Datasets/LN/annotations \
--coco_path ~/Datasets/coco
```

### Prepare Visual Genome (VG) TFRecords

You need to download Visual Genome following [Instructions in GRIT](https://github.com/JialianW/GRiT/blob/master/datasets/DATASETS.md#vg-dataset).

```shell
python third_party/py/scenic/projects/pixel_llm/tools/build_vg_tfrecord.py \
--input_json ~/Datasets/VisualGenome/annotations/test.json \
--image_path ~/Datasets/VisualGenome/VG_100K/ \
--output_path ~/Datasets/VisualGenome/tfrecords/test.tfrecord

python third_party/py/scenic/projects/pixel_llm/tools/build_vg_tfrecord.py \
--input_json ~/Datasets/VisualGenome/annotations/train.json \
--image_path ~/Datasets/VisualGenome/VG_100K/ \
--output_path ~/Datasets/VisualGenome/tfrecords/train.tfrecord \
--num_shards 128
```

### Prepare Referring Expression Localization/Segmentation TFRecords

You need to download [the preprocessed MDETR-style json files](https://zenodo.org/records/10795249/files/refcoco_release.zip) and [UNINEXT json files](https://github.com/MasterBin-IIAU/UNINEXT/blob/master/assets/DATA.md#rec--res).

```shell
python scenic/projects/pixel_llm/tools/build_mdetr_ref_tfrecord.py \
--output_dir ~/Datasets/PixelLLM/mdetr_data
--ann_output_dir ~/Datasets/PixelLLM/mdetr_data/annotations
--coco_path ~/Projects/PixelLLM/coco/
--ref_anno_path ~/Projects/PixelLLM/MDETR/mdetr_annotations_with_mask
```

### Prepare LLaVA TFRecords

You need to follow [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) to download image and json files.

```shell
python scenic/projects/pixel_llm/tools/build_llava_tfrecord.py \
    --output_dir ~/Datasets/PixelLLM/llava/LLaVA-Instruct-150K \
    --input_json ~/Datasets/PixelLLM/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_root ~/Datasets/PixelLLM/llava_images
```
