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

"""List of data modalities which are used as feature dict keys."""

# Input image:
IMAGE = 'image'

# Image ID:
IMAGE_ID = 'image_id'

# Original size of the image before resize/pad:
ORIGINAL_SIZE = 'original_size'

# Bounding boxes of shape [num_instances, 4]:
BOXES = 'boxes'
PREDICTED_BOXES = 'pred_boxes'

# ID for each ground-truth box:
ANNOTATION_ID = 'annotation_id'

# Area of box:
AREA = 'area'

# Indicator whether a box contains a single instance (0) or a crowd/group (1):
CROWD = 'crowd'

# Pre-sigmoid logits of confidence values for predicted boxes.
LOGITS = 'pred_logits'

# Scores (confidences) between 0 and 1 for predicted boxes.
SCORES = 'scores'

# Mask indicating whether an instance is real (1) or padding (0):
INSTANCE_PADDING_MASK = 'instance_padding_mask'

# Per-instance integer labels (one label per instance):
INSTANCE_LABELS = 'instance_labels'

# Per-instance text labels (one label per instance):
INSTANCE_TEXT_LABELS = 'instance_text_labels'

# Per-instance multi-labels (multiple labels per instance, padded to
# [num_instances, max_num_labels]):
INSTANCE_MULTI_LABELS = 'instance_multi_labels'
INSTANCE_TEXT_MULTI_LABELS = 'instance_text_multi_labels'

# Per-image negative integer labels (classes that are not present in the image):
NEGATIVE_LABELS = 'negative_labels'

# Per-image negative text labels (classes that are not present in the image):
NEGATIVE_TEXT_LABELS = 'negative_text_labels'

# List of classes that are not exhaustively annotated in an image (e.g. LVIS):
NOT_EXHAUSTIVE_LABELS = 'not_exhaustive_labels'

# List of text queries:
TEXT_QUERIES = 'text_queries'

# List of tokenized text queries:
TEXT_QUERIES_TOKENIZED = 'text_queries_tokenized'
