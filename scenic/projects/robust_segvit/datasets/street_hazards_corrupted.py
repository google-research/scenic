"""Street Hazards Corrupted Dataset."""

import os

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

_CORRUPTIONS = [
    'gaussian_noise',
    'brightness',
    'contrast',
    'fog',
]

def _make_builder_configs():
  """Construct a list of BuilderConfigs.

  Construct a list of 95 Cifar10CorruptedConfig objects, corresponding to
  the 15 corruption types + 4 extra corruptions and 5 severities.

  Returns:
    A list of StreetHazardsCorruptedConfig objects.
  """
  config_list = []
  for corruption in _CORRUPTIONS:
    for severity in range(1, 6):
      config_list.append(
          StreetHazardsCorruptedConfig(
              corruption_type=corruption,
              severity=severity,
              name="street_hazards_{}_{}".format(corruption, str(severity)),
              description='street_hazards. Corruption method: ' + corruption +
              ', severity level: ' + str(severity),
          ))
  return config_list


class StreetHazardsCorruptedConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Street Hazards corrupted.

    Args:
      corruption_type (str): name of corruption.
      severity (int): level of corruption.
      right_images (bool): Enables right images for stereo image tasks.
      segmentation_labels (bool): Enables image segmentation labels.
      disparity_maps (bool): Enables disparity maps.
      train_extra_split (bool): Enables train_extra split. This automatically
        enables coarse grain segmentations, if segmentation labels are used.
  """

  def __init__(self,
               *,
               corruption_type,
               severity,
               **kwargs):
    super(StreetHazardsCorruptedConfig, self).__init__(version='1.0.0', **kwargs)

    self.corruption = corruption_type
    self.severity = severity


class StreetHazardsCorrupted(tfds.core.GeneratorBasedBuilder):
  """Base class for StreetHazard dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Download files from _DOWNLOAD_URL and place them in the manual directory
  """

  VERSION = tfds.core.Version("0.0.0")
  BUILDER_CONFIGS = _make_builder_configs()
  RELEASE_NOTES = {
      '0.0.0': 'Street Hazards corruptions',
  }
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
    dl_paths = {}
    dl_paths['images'] = os.path.join(dl_manager._extract_dir, 'street_hazards_{}-{}/train'.format(self.builder_config.corruption, self.builder_config.severity))
    dl_paths['annotations'] = os.path.join(dl_manager._extract_dir, 'street_hazards/train')
    dl_paths['test_images'] = os.path.join(dl_manager._extract_dir, 'street_hazards_{}-{}/test'.format(self.builder_config.corruption, self.builder_config.severity))
    dl_paths['test_annotations'] = os.path.join(dl_manager._extract_dir, 'street_hazards/test')

    if any(not tf.io.gfile.exists(z) for z in dl_paths.values()):
      msg = 'You must download the dataset files manually and place them in: '
      msg += ', '.join(dl_paths.values())
      raise AssertionError(msg)

    return [
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
