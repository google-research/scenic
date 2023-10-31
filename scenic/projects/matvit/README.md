## Matryoshka Vision Transformers (MatViT)

This project implements MatViT in Scenic.

This is implemented according to the MatViT adaptation in the MatFormer paper, for details see
Devvrit et al., "[MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707)". This implementation provides the basic proof of concept and further optimizations can be applied to speed up the training.

### Training MatViT
The following command trains the MatViT-B/16 model


```python
$ python -m scenic.projects.matvit.main \
  --config=scenic/projects/matvit/configs/imagenet_augreg_matvit_config.py \
  --workdir=matvit_b_16/
```

### Evaluation Mix'n'match ImageNet-1k Validation Accuracy
The following command evaluates the MatViT-B/16 model on full MLP dimension (3072d)


```python
$ python -m scenic.projects.matvit.classification_eval_main \
  --model_path=${PATH_TO_MODEL} \
  --matvit_dims="3072,3072,3072,3072,3072,3072,3072,3072,3072,3072,3072,3072"
```

### Checkpoints

We made MatViT-B/16 and MatViT-L/16 models available in the following links

|  Model Size  |  Download Link  |  Training Details |
|:----------:|:------------:|:------------------------:|
| MatViT-B/16 | https://storage.googleapis.com/scenic-bucket/matvit/MatViT-B16-IN1K | ImageNet-1k |
| MatViT-L/16 | https://storage.googleapis.com/scenic-bucket/matvit/MatViT-L16-IN21K%2BIN1K | ImageNet-21k pre-training + ImageNet-1k finetuning |


If you are interested in using the code, please contact [Kaifeng Chen](mailto:francischen@google.com) and [Aditya Kusupati](mailto:kusupati@google.com).
