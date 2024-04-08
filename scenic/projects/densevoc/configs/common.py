# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common variables for Dense VOC."""
# pylint: disable=line-too-long

# Download from
#   https://huggingface.co/google-bert/bert-base-uncased/blob/main/vocab.txt
BERT_TOKENIZER_PATH = '/path/to/bert-base-uncased/vocab.txt'

NUM_EXAMPLES_SMIT_TRAIN = 481094
NUM_EXAMPLES_COCO_TRAIN = 118287
NUM_VG_TRAIN_EXAMPLES = 77396
NUM_VG_VAL_EXAMPLES = 5000
NUM_VIDSTG_TRAIN_VIDEOS = 5436
NUM_VIDSTG_VAL_VIDEOS = 603
NUM_VIDSTG_VAL_FPS1_MAX40F_IMAGES = 16434
NUM_VLN_TRAIN_VIDEOS = 5136
NUM_VLN_VAL_SEGMENTS = 2451

Build VG tf records using `tools/build_vg_tfrecord.py`
VG_TRAIN_PATH = '/path/to/vg/tfrecords/train.tfrecord@128'
VG_TEST_PATH = '/path/to/vg/tfrecords/test.tfrecord'
VG_TEST_ANN_PATH = '/path/to/vg/annotations/test.json'
# Build SMiT tfrecord using `tools/build_smit_tfrecord.py`
SMIT_TRAIN_PATH = '/path/to/smit_train.tfrecord@1024'
# Build VidSTG tfrecord using `tools/build_vidstg_tfrecord.py`
VIDSTG_TRAIN_VIDEO_TFRECORD_PATH = '/path/to/vidstg.video.caption.train.tfrecord@256'
VIDSTG_VAL_VIDEO_TFRECORD_PATH = '/path/to/vidstg.video.max200f.caption.val.tfrecord@32'
# Build VidSTG image tfrecord in tools/convert_video_tfrecord_to_image_tfrecord.py
VIDSTG_VAL_IMAGE_TFRECORD_PATH = '/path/to/vidstg.image.max200f.caption.val.tfrecord@32'
# Create the coco format json using `tools/create_coco_json_from_tfrecord.py`
VIDSTG_VAL_VIDEO_ANN_PATH = '/path/to/vidstg_max200f_val_coco_format.json'
VIDSTG_VAL_IMAGE_ANN_PATH = VIDSTG_VAL_VIDEO_ANN_PATH
# Build VLN tfrecord using `tools/build_vng_tfrecord.py`
VLN_UVO_TRAIN_PATH = '/path/to/vng_uvo_sparse_train.tfrecord@32'
VLN_UVO_VAL_PATH = '/path/to/vng_uvo_sparse_val.tfrecord@32'
# Create the coco format json using `tools/create_coco_json_from_tfrecord.py`
VLN_UVO_VAL_ANN_PATH = '/path/to/vng_uvo_sparse_val_coco_format.json'
# Convert CLIP weights using `tools/densevoc_convert_clip_b16_weights_to_jax.ipynb`
CLIP_WEIGHT_PATH = '/path/to/clip_b_16/'
