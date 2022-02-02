"""DMVR input pipeline utilities for temporal regression datasets.
"""
from typing import Dict, Optional

from dmvr import builders
from dmvr import processors

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def sample_start_and_goal(
    sequence: Dict[str, tf.Tensor],
    history_length: int = 1,
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    seed: Optional[int] = None,
    ) -> Dict[str, tf.Tensor]:
  """Sample start and goal frames from the sequence."""
  sequence_length = tf.shape(input=sequence[img_feature_name])[0]
  # Draw each unique start-end frame pair at uniform: implemented as weighted
  # sampling of the start frame (relative to how many potential end frames
  # follow it) and uniform sampling of the end frame.

  num_successors = tf.range(sequence_length - 1, 0, -1)
  # {1, ..., sequence_length - 1}: sequence_length - 1 items with mean
  # (1 + sequence_length - 1) / 2
  num_pairs = (sequence_length - 1) * sequence_length / 2
  probs = tf.cast(num_successors, tf.float32) / tf.cast(num_pairs, tf.float32)
  start_dist = tfd.Categorical(probs=probs)
  start = start_dist.sample()

  goal = tf.random.uniform(
      (1,),
      minval=tf.cast(start + 1, dtype=tf.int32),
      maxval=tf.cast(sequence_length, dtype=tf.int32),
      dtype=tf.int32,
      seed=seed)

  indices = tf.concat(
      [tf.maximum(0, tf.range(start - history_length + 1, start + 1)),
       goal], axis=0)
  indices.set_shape((history_length + 1,))
  frames = tf.gather(sequence[img_feature_name], indices)
  sequence[img_feature_name] = frames
  sequence['targets'] = goal - start
  return sequence


def sample_start(
    sequence: Dict[str, tf.Tensor],
    history_length: int = 1,
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    seed: Optional[int] = None,
    ) -> Dict[str, tf.Tensor]:
  """Sample start frame from the sequence, treating the last image as a goal."""
  sequence_length = tf.shape(input=sequence[img_feature_name])[0]

  start = tf.random.uniform(
      (),
      minval=tf.cast(0, dtype=tf.int32),
      maxval=tf.cast(sequence_length - 1, dtype=tf.int32),
      dtype=tf.int32,
      seed=seed)
  goal = tf.convert_to_tensor([sequence_length - 1], dtype=tf.int32)

  indices = tf.concat(
      [tf.maximum(0, tf.range(start - history_length + 1, start + 1)),
       goal], axis=0)
  indices.set_shape((history_length + 1,))
  frames = tf.gather(sequence[img_feature_name], indices)
  sequence[img_feature_name] = frames
  sequence['targets'] = goal - start
  return sequence


def sample_frames(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 3,
    stride: int = 1,
    min_resize: int = 224,
    crop_size: int = 200,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    augment_goals: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False):
  """Adds functions to process start and goal images to builders."""
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # TODO(minttu): Support stride.
  if stride != 1:
    raise NotImplementedError('stride != 1 not supported.')
  # Num_frames includes goal frame.
  if augment_goals:
      # pylint: disable=g-long-lambda
    sampler_builder.add_fn(
        fn=lambda x: sample_start_and_goal(x, num_frames - 1,
                                           output_feature_name),
        fn_name=f'{output_feature_name}_random_goal_sample',)
      # pylint: enable=g-long-lambda
  else:
    sampler_builder.add_fn(
        fn=lambda x: sample_start(x, num_frames - 1, output_feature_name),
        fn_name=f'{output_feature_name}_last_goal_sample',)

  # Decode JPEG string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the preprocessing.
  # Hence the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      fn=lambda x: processors.resize_smallest(x, min_resize, is_flow=is_flow),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')

  if is_training:
    # Note: Random flip can be problematic for tasks with left-right asymmetry,
    # e.g. "push something from left to right".
    # Standard image data augmentation: random crop and random flip.
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.crop_image(
            x, crop_size, crop_size, True, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_crop',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.random_flip_left_right(
            x, state=s, is_flow=is_flow),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_flip',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    # Central crop of the frames.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.crop_image(x, crop_size, crop_size, False),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_central_crop')

  # Cast the frames to `tf.float32`, normalizing according to
  # `zero_centering_image`.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.normalize_image(x, zero_centering_image),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_normalize')


