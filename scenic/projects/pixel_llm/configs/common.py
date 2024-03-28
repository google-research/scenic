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

# pylint: disable=line-too-long
"""Common utilities for config files."""

import collections

SP_VOCAB_SIZE = 32128  # sentence_piece tokenizer vocabulary used in T5
BERT_TOKENIZER_PATH = '/path/to/bert_tokenizer/'

TFRecordData = collections.namedtuple('TFRecordData', ['path', 'size'])

VG_TRAIN = TFRecordData(
    path='/path/to/vg_train',
    size=77396
)
VG_TEST = TFRecordData(
    path='/path/to/vg_test',
    size=5000
)
LN_COCO_TRAIN = TFRecordData(
    path='/path/to/ln_coco_train',
    size=118287
)
LN_COCO_VAL = TFRecordData(
    path='/path/to/ln_coco_val',
    size=5000
)
UNI_FLICKR_TRAIN = TFRecordData(
    path='/path/to/flickr_train',
    size=29778
)
UNI_MIXED_COCO_TRAIN = TFRecordData(
    path='/path/to/mixed_coco_train',
    size=28158,
)
UNI_MIXED_VG_TRAIN = TFRecordData(
    path='/path/to/mixed_vg_train',
    size=106635,
)
MERGE_COCO_IMAGE_SAFE_TRAIN = TFRecordData(
    path='/path/to/merge_coco_image_safe',
    size=24407,
)
REFCOCO_UNC_TRAIN = TFRecordData(
    path='/path/to/refcoco_unc_train',
    size=16994,
)
REFCOCOG_UMD_TRAIN = TFRecordData(
    path='/path/to/refcocog_umd_train',
    size=21899,
)
REFCOCOPLUS_UNC_TRAIN = TFRecordData(
    path='/path/to/refcocoplus_unc_train',
    size=16992,
)
REFCOCO_UNC_VALIDATION = TFRecordData(
    path='/path/to/refcoco_unc_validation',
    size=1500,
)
REFCOCO_UNC_TESTA = TFRecordData(
    path='/path/to/refcoco_unc_testa',
    size=750,
)
REFCOCO_UNC_TESTB = TFRecordData(
    path='/path/to/refcoco_unc_testb',
    size=750,
)
REFCOCOG_UMD_VALIDATION = TFRecordData(
    path='/path/to/refcocog_umd_validation',
    size=1300,
)
REFCOCOG_UMD_TEST = TFRecordData(
    path='/path/to/refcocog_umd_test',
    size=2600,
)
REFCOCOPLUS_UNC_VALIDATION = TFRecordData(
    path='/path/to/refcocoplus_unc_validation',
    size=1500,
)
REFCOCOPLUS_UNC_TESTA = TFRecordData(
    path='/path/to/refcocoplus_unc_testa',
    size=750,
)
REFCOCOPLUS_UNC_TESTB = TFRecordData(
    path='/path/to/refcocoplus_testb',
    size=750,
)


COCO_CAP_TEST_ANNOTATION='/path/to/coco_cap_test_annotation',

VG_DENSECAP_TEST_ANNOTATION='/path/to/vg_densecap_test_annotation',

REFCOCO_UNC_VALIDATION_ANNOTATION='/path/to/refcoco_unc_validation_annotation',

REFCOCO_UNC_TESTA_ANNOTATION='/path/to/refcoco_unc_testa_annotation',

REFCOCO_UNC_TESTB_ANNOTATION='/path/to/refcoco_unc_testb_annotation',

REFCOCOG_UMD_VALIDATION_ANNOTATION='/path/to/refcocog_umd_validation_annotation',

REFCOCOG_UMD_TEST_ANNOTATION='/path/to/refcocog_umd_test_annotation',

REFCOCOPLUS_UNC_VALIDATION_ANNOTATION='/path/to/refcocoplus_unc_validation_annotation',

REFCOCOPLUS_UNC_TESTA_ANNOTATION='/path/to/refcocoplus_unc_testa_annotation',

REFCOCOPLUS_UNC_TESTB_ANNOTATION='/path/to/refcocoplus_unc_testb_annotation',
