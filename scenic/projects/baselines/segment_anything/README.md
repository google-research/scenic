Segment Anything
==

Scenic re-implementation of
[Segment Anything](https://ai.facebook.com/research/publications/segment-anything/),
which is a zero-shot segmentation module that takes an image and a prompt as
input and produce segmentation mask of the indicated object.

 - Paper: Segment Anything, Kirillov et al. ICCV 2023
 - Reference pytorch code: https://github.com/facebookresearch/segment-anything

## Setup

Run [this colab](notebooks/Convert_SAM_weights.ipynb) to convert the official
PyTorch pretrained weights to JAX.
Download the converted weights to a local path.

## Usage:

```
from flax.training import checkpoints
from scenic.projects.baselines.segment_anything import demo_utils
from scenic.projects.baselines.segment_anything.modeling import sam


# setup model and load weights
input_size = 1024
model_size = 'B'
checkpoint_path = '/path/to/sav_vit_x/'
sam_model = sam.Sam(
    image_encoder_args=demo_utils.get_encoder_config(model_size))
params = checkpoints.restore_checkpoint(checkpoint_path, None)['params']

# Load a test image
image_path = '/path/to/image/'
image = demo_utils.load_image(image_path)
input_image, padding_mask, ori_size = demo_utils.resize_and_pad_image(
    image, target_size=input_size)

## Prompt-based segmentation
# Prepare point prompts
point_prompts = [[500, 375]]
point_coords, point_labels = demo_utils.get_point_coords_and_labels(
    point_prompts, input_size, ori_size,
)

# Run model
ret = sam_model.apply(
    {'params': params},
    input_image,
    point_coords,
    point_labels,
    padding_mask,
    return_image_embedding=True,
    train=False)

# Visualize outputs
for mask, score in zip(ret[0]['masks'][0], ret[0]['iou_predictions'][0]):
  demo_utils.plot(input_image[0], point_coords[0], point_labels[0], mask, score)

# To run the model again with more prompts using the cached image embeddings
# from the previous run.
cached_image_embedding = ret[0]['image_embedding'][None]
ret = sam_model.apply(
    {'params': params},
    input_image=None,
    point_coords=point_coords,
    point_labels=point_labels,
    padding_mask=padding_mask,
    image_embeddings=cached_image_embedding,
    train=False)

## Segment all objects

# Run model
ret = sam_model.apply(
    {'params': params},
    image=input_image[0],
    padding_mask=padding_mask[0],
    method=sam_model.generate)

# Visualize outputs
demo_utils.plot_all_masks(input_image[0], ret)

```
