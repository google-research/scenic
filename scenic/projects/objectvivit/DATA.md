# Setting up datasets for ObjectViViT

Our dataloader is based on
[DeepMind Video Reader](https://github.com/deepmind/dmvr) which uses
[TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).

First, follow the [data instruction in ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit/data)
to create the TFRecords for the [SomethingSomethingV2](https://developer.qualcomm.com/software/ai-datasets/something-something) dataset.
Then we'll add bounding box data to the TFRecords.
We use bounding boxes from [ORViT](https://github.com/eladb3/ORViT).
Download the ORViT bounding boxes [here](https://github.com/eladb3/ORViT/blob/master/slowfast/datasets/DATASET.md#something-something-v2)
and unzip it, then run the following script to add them to TFRecords:

```
python scenic/projects/objectvivit/tools/add_orvit_bbox_to_tfrecord.py \
--input_tfrecord /path/to/ssv2/tfrecord@xxx \
--bbox_folder /path/to/orvid/box/folder/ \
--output_tfrecord  /path/to/ssv2.orvit_box/tfrecord
```

Finally, update the data path `/path/to/ssv2.orvit_box/tfrecord` in the [config files](scenic/projects/objectvivit/configs).
