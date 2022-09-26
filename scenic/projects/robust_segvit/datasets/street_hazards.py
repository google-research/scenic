"""Street Hazards Dataset."""

import os
import re

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@article{hendrycks2019anomalyseg,
  title={Scaling Out-of-Distribution Detection for Real-World Settings},
  author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and Zou, Andy and Kwon, Joe and Mostajabi, Mohammadreza and Steinhardt, Jacob and Song, Dawn},
  journal={ICML},
  year={2022}
}
"""

_DESCRIPTION = """\
Streethazards dataset.
The original dataset can be downloaded from:
https://github.com/hendrycks/anomaly-seg

Then unzip the file and place it in the following directory:
tensorflow_datasets/downloads/extracted/streethazard
"""

_DOWNLOAD_URL = "gs://ub-ekb/streethazard/raw_data/v.0.0"

_TRAIN_URL = {
    "images":
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar",
    "annotations":
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar"
}

_TEST_URL = {
    "images":
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar",
    "annotations":
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar"
}

class StreetHazards(tfds.core.GeneratorBasedBuilder):
  """Base class for StreetHazard dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Download files from _DOWNLOAD_URL and place them in the manual directory
  """

  VERSION = tfds.core.Version("0.0.0")

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(encoding_format="png"),
            "annotations": tfds.features.Image(encoding_format="png")
        }),
        supervised_keys=("image", "annotations"),
        homepage="https://github.com/hendrycks/anomaly-seg",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    import pdb; pdb.set_trace()
    dl_paths = {}
    dl_paths['images'] = os.path.join(dl_manager._extract_dir, 'street_hazards/train')
    dl_paths['annotations'] = os.path.join(dl_manager._extract_dir, 'street_hazards/train')
    dl_paths['test_images'] = os.path.join(dl_manager._extract_dir, 'street_hazards/test')
    dl_paths['test_annotations'] = os.path.join(dl_manager._extract_dir, 'street_hazards/test')

    if any(not tf.io.gfile.exists(z) for z in dl_paths.values()):
      msg = 'You must download the dataset files manually and place them in: '
      msg += ', '.join(dl_paths.values())
      raise AssertionError(msg)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images_dir_path":
                    os.path.join(dl_paths["images"], "images/training/t1-3"),
                "annotations_dir_path":
                    os.path.join(dl_paths["annotations"],
                                 "annotations/training/t1-3")
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "images_dir_path":
                    os.path.join(dl_paths["images"], "images/validation/t4"),
                "annotations_dir_path":
                    os.path.join(dl_paths["annotations"],
                                 "annotations/validation/t4")
            },
        ),

      # both t5 and t6
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "images_dir_path":
            os.path.join(dl_paths["test_images"], "images/test/t5-6"),
          "annotations_dir_path":
            os.path.join(dl_paths["test_annotations"],
                         "annotations/test/t5-6")
        },
      ),
    ]

  def _generate_examples(self, images_dir_path, annotations_dir_path):
      for image_file in tf.io.gfile.listdir(images_dir_path):
          # get the filename
          image_id = os.path.split(image_file)[1].split(".")[0]
          yield image_id, {
              "image":
                  os.path.join(images_dir_path, "{}.png".format(image_id)),
              "annotations":
                  os.path.join(annotations_dir_path, "{}.png".format(image_id))
          }
