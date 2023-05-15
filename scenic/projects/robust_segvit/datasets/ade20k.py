"""ADE20k-Dataset."""

import os
import re

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{zhou2017scene,
title={Scene Parsing through ADE20K Dataset},
author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year={2017}
}

"""

_DESCRIPTION = """\
ADE20K dataset.
The original dataset can be downloaded from:
http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

Then unzip the file and place it in the following directory:
tensorflow_datasets/downloads/extracted/ADEChallengeData2016
"""

_DOWNLOAD_URL = "gs://ub-ekb/ade20k/raw_data/v.0.0"

_TRAIN_URL = {
    "images":
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016",
    "annotations":
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016"
}


class ADE20k(tfds.core.GeneratorBasedBuilder):
  """Base class for ADE20k dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Download files from _DOWNLOAD_URL and place them in the manual directory
  """

  VERSION = tfds.core.Version("0.0.0")

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(encoding_format="jpeg"),
            "annotations": tfds.features.Image(encoding_format="png")
        }),
        supervised_keys=("image", "annotations"),
        homepage="http://sceneparsing.csail.mit.edu/",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    dl_paths = {}
    dl_paths['images'] = os.path.join(dl_manager._extract_dir, os.path.basename(_TRAIN_URL['images']))
    dl_paths['annotations'] = os.path.join(dl_manager._extract_dir, os.path.basename(_TRAIN_URL['annotations']))

    if any(not tf.io.gfile.exists(z) for z in dl_paths.values()):
      msg = 'You must download the dataset files manually and place them in: '
      msg += ', '.join(dl_paths.values())
      raise AssertionError(msg)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images_dir_path":
                    os.path.join(dl_paths["images"], "images/training"),
                "annotations_dir_path":
                    os.path.join(dl_paths["annotations"],
                                 "annotations/training")
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "images_dir_path":
                    os.path.join(dl_paths["images"], "images/validation"),
                "annotations_dir_path":
                    os.path.join(dl_paths["annotations"],
                                 "annotations/validation")
            },
        ),
    ]

  def _generate_examples(self, images_dir_path, annotations_dir_path):
      for image_file in tf.io.gfile.listdir(images_dir_path):
          # get the filename
          image_id = os.path.split(image_file)[1].split(".")[0]
          yield image_id, {
              "image":
                  os.path.join(images_dir_path, "{}.jpg".format(image_id)),
              "annotations":
                  os.path.join(annotations_dir_path, "{}.png".format(image_id))
          }
