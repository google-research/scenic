"""Fishyscapes Lost & Found Validation Dataset.

Edited from
https://github.com/hermannsblum/bdl-benchmark/blob/master/bdlb/fishyscapes/fishyscapes_tfds.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from itertools import chain
from os import path
import re
import numpy as np

import tensorflow_datasets as tfds
from tensorflow_datasets.image.lost_and_found import LostAndFound, LostAndFoundConfig
from tensorflow_datasets.image.cityscapes import Cityscapes, CityscapesConfig

_CITATION = """
@article{blum2019fishyscapes,
  title={The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation},
  author={Blum, Hermann and Sarlin, Paul-Edouard and Nieto, Juan and Siegwart, Roland and Cadena, Cesar},
  journal={arXiv preprint arXiv:1904.03215},
  year={2019}
}
"""

_DESCRIPTION = """
Benchmark of anomaly detection for semantic segmentation in urban driving images.
"""

_RELEASE_NOTES = """
3.0.0: january 2020: added cityscapes objects as negative test
2.0.0: june 2019: improved blending
1.0.0: march 2019: first version
"""

class FishyscapesConfig(tfds.core.BuilderConfig):
  '''BuilderConfig for Fishyscapes

    Args:
  '''
  def __init__(self, base_data='lost_and_found', original_mask=False, **kwargs):
      super().__init__(**kwargs)
      assert base_data in ['lost_and_found', 'cityscapes']
      self.base_data = base_data
      self.original_mask = original_mask


def _make_builder_configs():
  """Construct a list of BuilderConfigs."""

  BUILDER_CONFIGS = [
    FishyscapesConfig(
        name='LostAndFound',
        description='Validation set based on LostAndFound images.',
        version=tfds.core.Version('1.0.0'),
        base_data='lost_and_found',
        original_mask=False,
    ),
    FishyscapesConfig(
        name='OriginalLostAndFound',
        description='Validation set based on LostAndFound images.',
        version=tfds.core.Version('1.0.0'),
        base_data='lost_and_found',
        original_mask=True,
    ),
    FishyscapesConfig(
        name='Static',
        description='Validation set based on Cityscapes and Pascal VOC images.',
        version=tfds.core.Version('3.0.0'),
        supported_versions=[
            tfds.core.Version('2.0.0'),
            tfds.core.Version('1.0.0'),
        ],
        base_data='cityscapes',
    )]
  return BUILDER_CONFIGS

class Fishyscapes(tfds.core.GeneratorBasedBuilder):
  """Fishyscapes Lost & Found Validation Dataset"""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Download files from cityscapes and place them in the manual directory
  """

  BUILDER_CONFIGS = _make_builder_configs()
  RELEASE_NOTES = {
      '3.0.0': 'Fishyscapes',
  }

  def _info(self):
      return tfds.core.DatasetInfo(
          builder=self,
          # This is the description that will appear on the datasets page.
          description=_DESCRIPTION,
          # tfds.features.FeatureConnectors
          features=tfds.features.FeaturesDict({
              # These are the features of your dataset like images, labels ...
              'image_id': tfds.features.Text(),
              'basedata_id': tfds.features.Text(),
              'image_left': tfds.features.Image(shape=(1024, 2048, 3),
                                                encoding_format='png'),
              'mask': tfds.features.Image(shape=(1024, 2048, 1),
                                          encoding_format='png'),
          }),
          supervised_keys=('image_left', 'mask'),
          # Homepage of the dataset for documentation
          homepage='https://fishyscapes.com/',
          citation=_CITATION,
      )

  def _split_generators(self, dl_manager):
      """Returns SplitGenerators."""
      # download the data
      # TODO add the cityscapes overlays
      dl_paths = dl_manager.download({
          'lostandfound_mask': 'http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fishyscapes_lostandfound.zip',
          'cityscapes_overlays_v1': 'http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fs_val_v1.zip',
          'cityscapes_overlays_v2': 'http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fs_val_v2.zip',
          'cityscapes_overlays_v3': 'http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fs_val_v3.zip',
      })
      dl_paths = dl_manager.extract(dl_paths)

      # only way to get the tfds downlaod path
      download_dir = path.join(self._data_dir_root, 'downloads')
      if self.builder_config.base_data == 'lost_and_found':
          base_builder = LostAndFound(config=LostAndFoundConfig(
              name='fishyscapes',
              description='Config to generate images for the Fishyscapes dataset.',
              version='1.1.0',
              right_images=False,
              segmentation_labels=True,
              instance_ids=False,
              disparity_maps=False,
              use_16bit=False))
          downloaded_data = dl_paths['lostandfound_mask']
          base_dl_manager = dl_manager
      elif self.builder_config.base_data == 'cityscapes':
          base_builder = Cityscapes(config='semantic_segmentation')
          if self.builder_config.version == '1.0.0':
              downloaded_data = dl_paths['cityscapes_overlays_v1']
          elif self.builder_config.version == '2.0.0':
              downloaded_data = dl_paths['cityscapes_overlays_v2']
          elif self.builder_config.version == '3.0.0':
              downloaded_data = dl_paths['cityscapes_overlays_v3']
          base_dl_manager = tfds.download.DownloadManager(
            manual_dir_instructions="Download dataset manually.",
            download_dir=download_dir,
            manual_dir=path.join(download_dir, 'manual/cityscapes'),)
      else:
          raise UserWarning('config contains unsupported base_data')
      # manually force a download and split generation for the base dataset
      # There is no tfds-API that allows for getting images by id, so this is the only
      # option.
      splits = base_builder._split_generators(base_dl_manager)
      generators = [base_builder._generate_examples(**split.gen_kwargs)
                    for split in splits]
      return [
          tfds.core.SplitGenerator(
              name=tfds.Split.VALIDATION,
              # These kwargs will be passed to _generate_examples
              gen_kwargs={'fishyscapes_path': downloaded_data,
                          'base_images': {key: features for key, features in chain(*generators)}},
          ),
      ]

  def _generate_examples(self, fishyscapes_path, base_images):
      """Yields examples."""
      for filename in tf.io.gfile.listdir(fishyscapes_path):
          if filename.endswith('_labels.png'):
              fs_id, cityscapes_id = _get_ids_from_labels_file(filename)
              features = {
                'image_id': fs_id,
                'basedata_id': cityscapes_id,
                'mask': path.join(fishyscapes_path, filename),
              }
              if self.builder_config.base_data == 'lost_and_found':
                  features['image_left'] = base_images[cityscapes_id]['image_left']
                  if self.builder_config.original_mask:
                      features['mask'] = base_images[cityscapes_id]['segmentation_label']
              elif self.builder_config.base_data == 'cityscapes':
                  overlay_image = next(f for f in tf.io.gfile.listdir(fishyscapes_path)
                                       if f.startswith(fs_id) and f.endswith('rgb.npz'))
                  overlay_image = np.load(
                    path.join(fishyscapes_path, overlay_image))['rgb'].astype(int)
                  base_image = base_images[cityscapes_id]['image_left']
                  base_image = tf.image.decode_jpeg(tf.io.read_file(base_image), channels=3)
                  base_image = np.array(base_image).astype(int)
                  features['image_left'] = np.clip(base_image + overlay_image, 0, 255).astype('uint8')
              yield fs_id, features

# Helper functions

IDS_FROM_FILENAME = re.compile(r'([0-9]+)_(.+)_labels.png')

def _get_ids_from_labels_file(labels_file):
    '''Returns the ids (fishyscapes and cityscapes format) from the filename of a labels
  file. Used to associate a fishyscapes label file with the corresponding cityscapes
  image.

  Example:
    '0000_04_Maurener_Weg_8_000000_000030_labels.png' -> '0000', '04_Maurener_Weg_8_000000_000030'
  '''
    match = IDS_FROM_FILENAME.match(labels_file)
    return match.group(1), match.group(2)
