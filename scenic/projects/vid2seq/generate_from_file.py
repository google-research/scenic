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

# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python script to generate TFRecords of SequenceExample from csv."""

import contextlib
import math
import os
from typing import Optional, Sequence

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

flags.DEFINE_string("csv_path", None, "Input csv")
flags.DEFINE_string("output_path", None, "Tfrecords output path.")
flags.DEFINE_string(
    "features_path",
    None,
    "In case features are stored in individual files and not in the csv.",
)
flags.DEFINE_integer(
    "num_shards",
    -1,
    (
        "Number of shards to output, -1 means"
        "it will automatically adapt to the sqrt(num_examples)."
    ),
)
flags.DEFINE_bool("shuffle_csv", False, "Whether or not to shuffle the csv.")
FLAGS = flags.FLAGS


@contextlib.contextmanager
def _close_on_exit(writers):
  """Call close on all writers on exit."""
  try:
    yield writers
  finally:
    for writer in writers:
      writer.close()


def add_float_list(key: str, values: Sequence[float],
                   sequence: tf.train.SequenceExample):
  sequence.feature_lists.feature_list[key].feature.add(
  ).float_list.value[:] = values


def add_bytes_list(key: str, values: Sequence[bytes],
                   sequence: tf.train.SequenceExample):
  sequence.feature_lists.feature_list[key].feature.add(
      ).bytes_list.value[:] = values


def add_int_list(key: str, values: Sequence[int],
                 sequence: tf.train.SequenceExample):
  sequence.feature_lists.feature_list[key].feature.add(
  ).int64_list.value[:] = values


def set_context_int_list(key: str, value: Sequence[int],
                         sequence: tf.train.SequenceExample):
  sequence.context.feature[key].int64_list.value[:] = value


def set_context_bytes(key: str, value: bytes,
                      sequence: tf.train.SequenceExample):
  sequence.context.feature[key].bytes_list.value[:] = (value,)


def set_context_float(key: str, value: float,
                      sequence: tf.train.SequenceExample):
  sequence.context.feature[key].float_list.value[:] = (value,)


def set_context_int(key: str, value: int, sequence: tf.train.SequenceExample):
  sequence.context.feature[key].int64_list.value[:] = (value,)


def generate_sequence_example(video_id: str,
                              start: Optional[Sequence[float]],
                              end: Optional[Sequence[float]],
                              caption: Optional[Sequence[str]],
                              asr_start: Sequence[float],
                              asr_end: Sequence[float],
                              asr_string: Sequence[str],
                              features: Sequence[Sequence[float]],
                              duration: int,
                              split: Sequence[int] = None):
  """Generate a sequence example."""

  # Initiate the sequence example.
  seq_example = tf.train.SequenceExample()

  # Add dense captioning annotations if these exist.
  if caption is not None:
    for s, e, c in zip(start, end, caption):
      seq_example.context.feature[
          "video/timestamps/start"
      ].int64_list.value.append(s)
      seq_example.context.feature[
          "video/timestamps/end"
      ].int64_list.value.append(e)
      seq_example.context.feature["caption/string"].bytes_list.value.append(
          c.encode()
      )

  # Add ASR.
  if asr_start:
    for s, e, c in zip(asr_start, asr_end, asr_string):
      seq_example.context.feature[
          "ASR/timestamps/start"
      ].int64_list.value.append(s)
      seq_example.context.feature["ASR/timestamps/end"].int64_list.value.append(
          e
      )
      seq_example.context.feature["ASR/string"].bytes_list.value.append(
          c.encode()
      )

  # Add visual features.
  for f in features:
    add_float_list("image/clip_embeddings", f, seq_example)

  if split is not None:
    for s in split:
      seq_example.context.feature["split"].int64_list.value.append(s)

  # Add other metadata.
  set_context_bytes("videoid", video_id.encode(), seq_example)
  set_context_int("video/duration", duration, seq_example)
  return seq_example


def main():
  # reads the input csv.
  input_csv = pd.read_csv(FLAGS.csv_path)
  if FLAGS.num_shards == -1:
    num_shards = int(math.sqrt(len(input_csv)))
  else:
    num_shards = FLAGS.num_shards
  # Set up the TFRecordWriters.
  basename = os.path.splitext(os.path.basename(FLAGS.csv_path))[0]
  shard_names = [
      os.path.join(FLAGS.output_path, f"{basename}-{i:05d}-of-{num_shards:05d}")
      for i in range(num_shards)
  ]
  writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]

  if FLAGS.shuffle_csv:
    input_csv = input_csv.sample(frac=1)
  with _close_on_exit(writers) as writers:
    for i in tqdm(range(len(input_csv))):
      print(
          "Processing example %d of %d   (%d%%) \r" %
          (i, len(input_csv), i * 100 / len(input_csv)),
          end="")
      if "caption" in input_csv:
        start = eval(input_csv["start"].values[i])  # pylint:disable=eval-used
        end = eval(input_csv["end"].values[i])  # pylint:disable=eval-used
        caption = eval(input_csv["caption"].values[i])  # pylint:disable=eval-used
      else:
        start = None
        end = None
        caption = None
      asr_start = input_csv["asr_start"].values[i]
      if isinstance(asr_start, str):
        asr_start = eval(asr_start)  # pylint:disable=eval-used
      asr_end = input_csv["asr_end"].values[i]
      if isinstance(asr_end, str):
        asr_end = eval(asr_end)  # pylint:disable=eval-used
      asr_string = input_csv["asr_string"].values[i]
      if isinstance(asr_string, str):
        asr_string = eval(asr_string)  # pylint:disable=eval-used
      video_id = input_csv["video_id"].values[i]
      split = None
      if "split" in input_csv:
        split = input_csv["split"].values[i]
      if isinstance(split, str):
        split = eval(split)  # pylint:disable=eval-used
      if "features" not in input_csv:  # load on the fly
        assert FLAGS.features_path
        features = list(
            np.load(os.path.join(FLAGS.features_path, video_id + ".npy"))
        )
      else:
        features = eval(input_csv["features"].values[i])  # pylint:disable=eval-used
      duration = int(input_csv["duration"].values[i])
      seq_ex = generate_sequence_example(
          video_id,
          start,
          end,
          caption,
          asr_start,
          asr_end,
          asr_string,
          features,
          duration,
          split)
      writers[i % len(writers)].write(seq_ex.SerializeToString())


if __name__ == "__main__":
  app.run(main)
