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

"""Preprocessing ops for text labels."""

import abc
import dataclasses
import functools
from typing import Optional, Sequence, Tuple, Union

from absl import logging
from clu import preprocess_spec
import numpy as np
from scenic.projects.baselines.detr import transforms as detr_transforms
from scenic.projects.owl_vit.clip import tokenizer as clip_tokenizer
from scenic.projects.owl_vit.preprocessing import image_ops
from scenic.projects.owl_vit.preprocessing import modalities
from scenic.projects.owl_vit.preprocessing import transforms

import tensorflow as tf
import tensorflow_datasets as tfds

Features = preprocess_spec.Features
all_ops = lambda: preprocess_spec.get_all_ops(__name__)

# Adding NOT_PROMPTABLE_MARKER to a query will exclude it from having a prompt
# template (e.g. 'a photo of a {}') added during training:
NOT_PROMPTABLE_MARKER = '#'

PADDING_QUERY = ''

# From
# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
CLIP_BEST_PROMPT_TEMPLATES = [
    'itap of a {}.',
    'a bad photo of the {}.',
    'a origami {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
]

# From
# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
CLIP_PAPER_PROMPT_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

TRAINING_PROMPT_TEMPLATES = ['{}'] + CLIP_PAPER_PROMPT_TEMPLATES

# From annotation JSON files at https://www.lvisdataset.org/dataset:
LVIS_RARE_CLASSES = [
    'applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax',
    'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo', 'barbell',
    'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper',
    'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp', 'gameboard',
    'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)',
    'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque', 'breechcloth',
    'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest',
    'burrito', 'cabana', 'locker', 'candy_bar', 'canteen', 'elevator_car',
    'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette',
    'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook',
    'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip',
    'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet',
    'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak',
    'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material',
    'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)',
    'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread',
    'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher',
    'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal',
    'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die',
    'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar',
    'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick',
    'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll',
    'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret',
    'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash',
    'fleece', 'football_helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
    'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish',
    'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat', 'griddle', 'grits',
    'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw',
    'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus',
    'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate',
    'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate',
    'inhaler', 'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard',
    'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala',
    'lab_coat', 'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume',
    'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun',
    'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher',
    'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake',
    'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile', 'neckerchief',
    'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)',
    'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose', 'papaya',
    'paperback_book', 'paperweight', 'parchment', 'passenger_ship',
    'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener',
    'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook',
    'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel',
    'tobacco_pipe', 'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)',
    'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding',
    'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla',
    'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)',
    'river_boat', 'road_map', 'rodent', 'roller_skate', 'Rollerblade',
    'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel',
    'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane',
    'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears',
    'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap',
    'shredder_(for_paper)', 'skullcap', 'sling_(bandage)', 'smoothie', 'snake',
    'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle',
    'sparkler_(fireworks)', 'spear', 'crawfish', 'squid_(food)', 'stagecoach',
    'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus',
    'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe', 'Tabasco_sauce',
    'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank',
    'telephoto_lens', 'tequila', 'thimble', 'trampoline', 'trench_coat',
    'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip',
    'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle_iron',
    'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf'
]

# The list below contains labels from Object365 and Visual Genome that are close
# to LVIS "rare" labels. Annotations with these labels must be removed from the
# training data for accurate "zero-shot" evaluation. The list was created by
# finding all O365/VG labels that contain LVIS labels as a substring (after
# removing space, underscore and dash). This catches close but non-identical
# labels such as "apple sauce" vs. "applesauce", "leather" vs "brown leather",
# or singular vs. plural. False positives were manually removed from the list.
O365_AND_VG_FORBIDDEN = [
    'apple cider', 'apple sauce', 'apricots', 'ax tool', 'axe', 'baguette',
    'baguettes', 'balsamic vinegar', 'barbell weights', 'barbells', 'barges',
    'baseball mascot', 'bbq cooker', 'beach ball', 'bean casserole',
    'bear mascot', 'bed pan', 'beef stew', 'beige fleece', 'big rat',
    'bird mascot', 'black fleece', 'black funnel', 'black garbage',
    'black leather', 'black leather corner', 'black pistol', 'black satchel',
    'blackleather', 'blue bonnet', 'blue pennant', 'blue plume', 'blue snake',
    'bobber', 'book mark', 'bookmarker', 'bookmarks', 'bottle liquor',
    'breakfast quiche', 'broken spear', 'brown gorilla', 'brown leather',
    'building gargoyle', 'burritos', 'cabana roof', 'cabanas', 'camera flash',
    'carnations', 'carrot stew', 'casserole dish', 'cassette disc',
    'cassette tape', 'cement cylinder', 'chaps', 'charcoal cooker',
    'check book', 'checker board', 'checkerboard pattern', 'chess board',
    'chime is hanging', 'chime is still', 'chimes', 'chocolate eclair',
    'clementines', 'clock pendulum', 'clothes hamper', 'coffee stirrer',
    'coil burner', 'coil heater', 'coil pipe', 'coil samples', 'coil wire',
    'coiled cable', 'coiled wire', 'coils', 'cooker plate', 'cooker unit',
    'cookers', 'cork board', 'corn bread', 'coveralls', 'crab meat', 'croutons',
    'cylinder figure', 'cylinder object', 'cylinders', 'cymbals',
    'dark leather', 'detergent bottle', 'diary cover', 'dish detergent',
    'dishwashing detergent', 'dog kennel', 'doll house', 'dollar bill',
    'dollars', 'doves', 'dragon fly', 'drum stick', 'drumsticks', 'dumb bell',
    'dust pan', 'ear plug', 'ear plugs', 'earplugs', 'egg casserole',
    'electric shears', 'electrical coil', 'exhaust funnel', 'eye patch',
    'fedora hat', 'fence kennel', 'fish bowl', 'flag pennants',
    'flash from camera', 'flashes', 'fleece jacket', 'fleece liner',
    'footlocker', 'fudge center', 'futon cushion', 'game board', 'garbage heap',
    'garbage pail', 'garbage pails', 'garbage pile', 'gargoyles', 'gas mask',
    'gemstones', 'glass cylinder', 'glass of lemonade', 'gold chime',
    'gorillas', 'gourds', 'grape popsicle', 'green fleece', 'green gourds',
    'green shawl', 'grey fleece', 'handcuffs', 'head jewels', 'head set',
    'headsets', 'heatin coil', 'hole puncher', 'hot plate', 'hot plates',
    'hour glass', 'house boat', 'iridescent shears', 'jewels', 'joysticks',
    'kegs', 'key card', 'kitchen shears', 'koala bear', 'laundry detergent',
    'laundry hamper', 'leather patch', 'leather satchel', 'leather square',
    'leather strip', 'legumes', 'liquor bottle', 'liquor bottles',
    'liquor spirit', 'liquorbottle', 'lockers', 'mascots', 'match box',
    'meat stew', 'metal shears', 'microphone headset', 'nail file',
    'nutcracker doll', 'omelet part', 'omelete', 'omelette', 'omeletter',
    'one dollar', 'paint scraper', 'panty hose', 'papayas', 'paper weight',
    'peg board', 'pencil sharpener', 'pendulums', 'pennant banner', 'pennants',
    'persimmons', 'phone book', 'pin wheel', 'pin wheels', 'pinwheels',
    'pistol in waistband', 'pitch fork', 'pitcher of lemonade', 'play snake',
    'polo mallet', 'poncho hood', 'potato masher', 'propane cylinder', 'prunes',
    'radar beacon', 'radar dish', 'radar equipment', 'red coils', 'red leather',
    'red poncho', 'red spear', 'redthimble', 'rice cooker', 'sand barge',
    'sauce pan', 'sauce pans', 'saw horse', 'saw horses', 'sawhorse bench',
    'sawhorses', 'scissors shears', 'scrape', 'sea plane', 'sheep shears',
    'silver armor', 'silver funnel', 'sketched handcuffs', 'skull cap',
    'sliced gourds', 'slow cooker', 'small baguette', 'spears',
    'spinach quiche', 'step ladder', 'stick mallet', 'stirrers',
    'storage locker', 'stuffed gorilla', 'sub woofer', 'tambourines',
    'tan leather', 'tangy lemonade', 'telephone books', 'there is an axe',
    'toy snake', 'trainstep ladder', 'turnip roots', 'turnips', 'tux jacket',
    'tuxedo', 'tuxedo jacket', 'tuxedos', 'two rats', 'vats', 'video cassettes',
    'vodka bottle', 'vodka bottles', 'vultures', 'wash basin', 'wash basins',
    'waste barge', 'white armor', 'white cylinder', 'white fleece',
    'white pegboard', 'white shears', 'wii joystick', 'wind chime',
    'wind chimes', 'windchime', 'windchimes', 'wolf head', 'wood armoire',
    'wooden axe', 'woolen fleece', 'yellow bulldozer'
]

PER_EXAMPLE_INSTANCE_MULTI_LABELS = 'per_example_instance_multi_labels'


@functools.lru_cache(maxsize=10)
def get_label_map(tfds_name: str, tfds_data_dir: Optional[str] = None):
  """Returns a {label: name} dict for a TFDS dataset."""
  try:
    builder = tfds.builder(tfds_name, data_dir=tfds_data_dir)
    label_names = ['padding'] + builder.info.features['objects']['label'].names
    return {i: name for i, name in enumerate(label_names)}
  except Exception:
    logging.info('Builder did not specify label names for %s', tfds_name)
    raise


def mark_not_promptable(x: tf.Tensor) -> tf.Tensor:
  """Marks a tensor of strings as not-promptable by appending a marker."""
  tf.debugging.Assert(
      tf.logical_not(
          tf.reduce_any(
              tf.strings.regex_full_match(x, f'.*{NOT_PROMPTABLE_MARKER}.*'))),
      data=[x],
      name='assert_promptability_marker_not_in_string')
  marked = tf.strings.join([tf.fill(tf.shape(x), NOT_PROMPTABLE_MARKER), x])
  # Never mark padding.
  return tf.where(tf.equal(x, PADDING_QUERY), PADDING_QUERY, marked)


def remove_promptability_marker(x: tf.Tensor) -> tf.Tensor:
  """Removes any promptability-marker-character from a tensor of strings."""
  return tf.strings.regex_replace(x, NOT_PROMPTABLE_MARKER, '')


def _canonicalize_string_tf(
    string: Union[str, Sequence[str], tf.Tensor]) -> tf.Tensor:
  """Brings text labels into a standard form."""

  string = tf.strings.lower(string)

  # Remove all characters that are not either alphanumeric, or dash, or space,
  # or NOT_PROMPTABLE_MARKER:
  string = tf.strings.regex_replace(
      string, f'[^a-z0-9-{NOT_PROMPTABLE_MARKER} ]', ' ')
  string = tf.strings.regex_replace(string, r'\s+', ' ')
  string = tf.strings.regex_replace(string, r'-+', '-')
  string = tf.strings.strip(string)

  # Remove characters that equal the promptability-maker but appear somewhere
  # other than the start of the string:
  string = tf.strings.regex_replace(
      string, f'([^^]){NOT_PROMPTABLE_MARKER}+', r'\1')

  return string


def _canonicalize_string_py(string: str) -> str:
  """Wraps _canonicalize_string_tf for Python strings."""
  return _canonicalize_string_tf(string).numpy().decode()


def _convert_tf_boxes_to_xyxy(bbox: tf.Tensor, image_size: Sequence[int]):
  """Convert yxyx [0, 1] normalized boxes to xyxy unnormalized format."""
  y0, x0, y1, x1 = tf.split(bbox, 4, axis=-1)
  h = tf.cast(image_size[0], tf.float32)
  w = tf.cast(image_size[1], tf.float32)

  y0 = tf.clip_by_value(y0 * h, 0.0, h)
  x0 = tf.clip_by_value(x0 * w, 0.0, w)
  y1 = tf.clip_by_value(y1 * h, 0.0, h)
  x1 = tf.clip_by_value(x1 * w, 0.0, w)

  bbox = tf.concat([x0, y0, x1, y1], axis=-1)
  return bbox


def _add_prompt(args):
  """Converts a single label name string to a prompt using a template."""
  text_label, prompt_template = args
  prompted = tf.strings.regex_replace(prompt_template, r'\{\}', text_label)
  # Prompts may introduce non-canonical formatting, so canonicalize again:
  return _canonicalize_string_tf(prompted)


def _sample_random_prompt_templates(num_samples: tf.Tensor,
                                    seed: tf.Tensor) -> tf.Tensor:
  """Returns num_samples prompt templates uniformly at random."""
  prompt_templates = tf.constant(TRAINING_PROMPT_TEMPLATES)
  num_prompt_templates = len(TRAINING_PROMPT_TEMPLATES)
  random_inds = tf.random.stateless_categorical(
      logits=tf.ones((1, num_prompt_templates)),
      num_samples=num_samples,
      seed=seed,
  )[0]
  return tf.gather(prompt_templates, random_inds)


class NamedPreprocessOp(abc.ABC, preprocess_spec.PreprocessOp):
  """Preprocessing base class that adds a name scope for easier debugging."""

  @abc.abstractmethod
  def apply(self, features: Features) -> Features:
    """Applies the op to the features."""
    pass

  def __call__(self, features: Features) -> Features:
    # Copy dict to avoid confusing in-place modification of inputs:
    features = dict(features)

    # Add name score so that runtime errors are easier to locate:
    with tf.name_scope(type(self).__name__):
      return self.apply(features)


@dataclasses.dataclass(frozen=True)
class DecodeVisualGenome:
  """Decoder class for visual genome dataset.

  Note that based on prior experiments, by default we are only using VG objects
  and not the regions.

  Attributes:
    include_objects: Whether VG objects should be included.
    include_regions: Whether VG regions should be included.
    is_promptable: Whether text labels should be treated as promptable.
    tfds_data_dir: Unused here. Used in input_pipeline.py.
  """

  include_objects: bool = True
  include_regions: bool = False
  is_promptable: bool = False
  tfds_data_dir: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    image = tf.cast(features['image'], tf.float32) / 255.0

    boxes, text_labels = [], []
    if self.include_objects:
      boxes.append(features['objects']['bbox'])  # float32, in range [0, 1].
      text_labels.append(features['objects']['name'])

    if self.include_regions:
      boxes.append(features['regions']['bbox'])  # float32, in range [0, 1].
      text_labels.append(features['regions']['phrase'])

    # Combined objects and regions.
    if boxes:
      boxes = tf.concat(boxes, axis=0)
      text_labels = tf.concat(text_labels, axis=0)
    else:
      raise ValueError('Either objects or regions should be included in VG.')

    # Remove empty text labels.
    # pylint: disable=g-explicit-bool-comparison
    boxes = boxes[text_labels != PADDING_QUERY]
    text_labels = text_labels[text_labels != PADDING_QUERY]
    # pylint: enable=g-explicit-bool-comparison

    # Visual Genome performs better without prompting:
    # First, remove any markers that might be present in the labels:
    text_labels = remove_promptability_marker(text_labels)
    if not self.is_promptable:
      text_labels = mark_not_promptable(text_labels)

    features_new = {
        modalities.IMAGE:
            image,
        modalities.BOXES:
            boxes,
        modalities.INSTANCE_TEXT_LABELS:
            text_labels,
        modalities.NEGATIVE_LABELS:
            tf.fill([1], -1),  # Dummy padding.
        modalities.NEGATIVE_TEXT_LABELS:
            tf.fill([1], PADDING_QUERY),  # Dummy padding.
        # There are no labels so everything just 0. Don't set to -1 as mosaic
        # operation will filter padding.
        modalities.INSTANCE_LABELS:
            tf.zeros_like(text_labels, dtype=tf.int32),
        modalities.CROWD:
            tf.zeros_like(text_labels, dtype=tf.int32),
    }

    if 'rng' in features:
      features_new[image_ops.SEED_KEY] = features['rng']
    return features_new


@dataclasses.dataclass(frozen=True)
class DecodeLvis(image_ops.DecodeLvisExample):
  is_promptable: bool = True
  tfds_data_dir: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    features = super().__call__(features)
    return IntegerToTextLabels(
        tfds_name='lvis', is_promptable=self.is_promptable)(features)


@dataclasses.dataclass(frozen=True)
class DecodeObjects365(image_ops.DecodeCocoExample):
  """Given an Object365 TFDS example, create features with boxes."""
  is_promptable: bool = True
  tfds_data_dir: Optional[str] = None

  def get_class_name(self, label_idx: tf.Tensor) -> tf.Tensor:
    """Reads and constructs a mapping from integer classes to text labels."""
    # First label is "padding" and needs to be removed:
    class_labels = list(get_label_map('objects365').values())[1:]
    classes = tf.convert_to_tensor(class_labels)
    return tf.gather(classes, label_idx)

  def __call__(self, features: Features) -> Features:
    features = features.copy()
    # Add missing field.
    features['objects']['id'] = tf.zeros_like(features['objects']['label'])
    features = super().__call__(features)

    # Dummy negative labels:
    features[modalities.NEGATIVE_LABELS] = tf.fill([1], -1)

    return IntegerToTextLabels(
        tfds_name='objects365', is_promptable=self.is_promptable)(features)


@dataclasses.dataclass
class CanonicalizeTextLabels(NamedPreprocessOp):
  """Removes non-alphanum chars (except promptability marker) from labels."""

  text_keys: Sequence[str] = (modalities.INSTANCE_TEXT_LABELS,
                              modalities.NEGATIVE_TEXT_LABELS)

  def apply(self, features: Features) -> Features:
    for text_key in self.text_keys:
      if text_key in features:
        features[text_key] = _canonicalize_string_tf(features[text_key])
    return features


@dataclasses.dataclass
class IntegerToTextLabels(NamedPreprocessOp):
  """Looks up class names from integer labels and adds them as text features.

    Attributes:
    tfds_name: The TFDS name of the dataset, used to determine the label map.
    tfds_data_dir: Optional custom data dir for non-standard TFDS datasets.
    is_promptable: Whether text labels should be treated as promptable.
    label_text_keys: A sequence of pairs of label ids and corresponding text
      labels. By default, the integer ids of INSTANCE_LABELS and NEGATIVE_LABELS
      are mapped to their string text label, and stored in INSTANCE_TEXT_LABELS
      and NEGATIVE_TEXT_LABELS, respectively.
  """

  tfds_name: str
  tfds_data_dir: Optional[str] = None
  is_promptable: bool = True
  label_text_keys: Sequence[Tuple[str, str]] = (
      (modalities.INSTANCE_LABELS, modalities.INSTANCE_TEXT_LABELS),
      (modalities.NEGATIVE_LABELS, modalities.NEGATIVE_TEXT_LABELS),
  )

  def _get_label_map(self):
    return get_label_map(self.tfds_name, self.tfds_data_dir)

  def __post_init__(self):
    self.label_map = self._get_label_map()
    assert self.label_map.get(0, PADDING_QUERY) in ['', 'pad', 'padding', 'N/A']
    self.label_map[0] = PADDING_QUERY

  def apply(self, features: Features) -> Features:
    for _, text_key in self.label_text_keys:
      if text_key in features:
        raise ValueError(f'{text_key} are already present in the features.')

    integer_labels = tf.constant(list(self.label_map.keys()))
    text_labels = tf.constant(list(self.label_map.values()))

    # This avoids using tf Lookup tables which are stateful, and break some
    # pipelines. It may not scale well as lookup time is linear in # elements.
    table = StatelessLookupTable(
        integer_labels, _canonicalize_string_tf(text_labels),
        default_value=tf.constant(PADDING_QUERY, tf.string))

    # Label maps start at 1.
    for label_key, text_key in self.label_text_keys:
      if label_key in features:
        features[text_key] = table.lookup(features[label_key] + 1)
        features[text_key] = remove_promptability_marker(features[text_key])
      if not self.is_promptable:
        features[text_key] = mark_not_promptable(features[text_key])

    return features


def _is_forbidden_label(labels: tf.Tensor) -> tf.Tensor:
  """Checks which elements of string tensor 'labels' are forbidden."""
  forbidden_labels = LVIS_RARE_CLASSES + O365_AND_VG_FORBIDDEN

  # Canonicalize both query and forbidden labels:
  forbidden_labels = _canonicalize_string_tf(forbidden_labels)
  labels = _canonicalize_string_tf(labels)

  # Remove dashes, which are not removed by _canonicalize_string and may differ
  # between query and forbidden labels:
  forbidden_labels = tf.strings.regex_replace(forbidden_labels, '-', '')
  labels = tf.strings.regex_replace(labels, '-', '')

  # Need unique set for tf.lookup.StaticHashTable:
  forbidden_labels, _ = tf.unique(forbidden_labels)

  forbidden_labels_table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          forbidden_labels, tf.ones_like(forbidden_labels, dtype=tf.bool)),
      default_value=False)
  return forbidden_labels_table.lookup(remove_promptability_marker(labels))


@dataclasses.dataclass
class RemoveForbiddenLabels(NamedPreprocessOp):
  """Removes annotations for classes that we want to evaluate zero-shot on.

  Currently, this means LVIS "rare" classes. Other classes are defined to be OK
  to appear in the training set and will not be considered strict "zero-shot"
  classes.

  Forbidden labels are removed also if they are marked "non-promptable".
  """

  instance_text_labels_key: str = modalities.INSTANCE_TEXT_LABELS
  negative_text_labels_key: str = modalities.NEGATIVE_TEXT_LABELS
  negative_labels_key: str = modalities.NEGATIVE_LABELS

  def apply(self, features: Features) -> Features:
    if self.instance_text_labels_key in features:
      keep = tf.logical_not(
          _is_forbidden_label(features[self.instance_text_labels_key]))
      for feature in image_ops.FEATURES_WITH_FIRST_INSTANCE_AXIS:
        if feature in features:
          features[feature] = features[feature][keep]

    if self.negative_text_labels_key in features:
      keep = tf.logical_not(
          _is_forbidden_label(features[self.negative_text_labels_key]))
      features[self.negative_text_labels_key] = features[
          self.negative_text_labels_key][keep]
      if self.negative_labels_key in features:
        features[self.negative_labels_key] = features[
            self.negative_labels_key][keep]

    return features


@dataclasses.dataclass
class AddRandomNegativeLabels(NamedPreprocessOp):
  """Adds randomly sampled labels as additional negative labels.

  This is similar to the Federated Loss proposed in
  https://arxiv.org/pdf/2103.07461.pdf, but samples negatives in proportion to
  their appearance in the dataset, rather than in proportion to the square root
  of their frequency (for simplicity).

  The op works by maintaining a queue of labels seen in the dataset. For each
  dataset example, a number of candidate labels are randomly drawn from the
  queue. Labels that do not appear as positives in the example are added to the
  negatives, up to total_num_negatives.

  To keep the queue full, all text labels of the example, and the candidate
  labels previously sampled from the queue, are enqueued back. After warmup,
  sampled labels from the queue will have the same distribution as in the
  dataset.

  If negative integer labels are present in the features, this op will remove
  them, because they are obsolete after adding randomly sampled negatives.

  Attributes:
    total_num_negatives: Random negatives will be added to the input features to
      bring the total number of negatives to total_num_negatives.
    queue_capacity: Maximal size of the label queue. On average, the queue size
      will be maintained at half of the maximum.
    queue: tf.queue.RandomShuffleQueue. Will be added automatically.
  """

  total_num_negatives: int = 50
  queue_capacity: int = 100_000
  queue: Optional[tf.queue.RandomShuffleQueue] = None

  def __post_init__(self):
    self.queue = tf.queue.RandomShuffleQueue(
        capacity=self.queue_capacity,
        min_after_dequeue=0,
        dtypes=[tf.string],
        shapes=[tf.TensorShape([])],
        shared_name='random_negatives_queue')
    # Initialize with empty strings:
    self.queue.enqueue_many(tf.constant([''] * self.queue_capacity))

  def apply(self, features: image_ops.Features) -> image_ops.Features:
    # Draw candidate negative labels:
    candidate_labels = self.queue.dequeue_many(self.total_num_negatives * 2)

    # Fill queue back up:
    labels_to_enqueue = tf.concat([
        features[modalities.INSTANCE_TEXT_LABELS],
        features[modalities.NEGATIVE_TEXT_LABELS],
        candidate_labels,
    ], axis=0)
    labels_to_enqueue = tf.boolean_mask(
        labels_to_enqueue, tf.not_equal(labels_to_enqueue, PADDING_QUERY),
        name='labels_to_enqueue')
    target_size = self.queue_capacity // 2
    needed_elements = tf.clip_by_value(
        target_size - self.queue.size(), 0, tf.size(labels_to_enqueue))
    enqueue_op = self.queue.enqueue_many(
        tf.slice(labels_to_enqueue, begin=[0], size=[needed_elements]))

    # Get negatives that are not in positives:
    with tf.control_dependencies([enqueue_op]):
      candidate_negatives = tf.sparse.to_dense(
          tf.sets.difference(
              candidate_labels[None, ...],
              features[modalities.INSTANCE_TEXT_LABELS][None, ...]))[0]

    # Set operations sort the labels alphabetically, so we shuffle again:
    candidate_negatives = tf.random.shuffle(candidate_negatives)

    # New negatives contain all the old negatives, plus randomly sampled ones,
    # up to total_num_negatives. In addition, we ensure that padding ('') is not
    # present by including it as first element before applying tf.unique and
    # then slicing it off:
    new_negatives = tf.concat([
        tf.constant([PADDING_QUERY]),
        features[modalities.NEGATIVE_TEXT_LABELS],
        candidate_negatives,
    ], axis=0)
    orig_num_negatives = tf.shape(features[modalities.NEGATIVE_TEXT_LABELS])[0]
    new_num_negatives = tf.maximum(self.total_num_negatives, orig_num_negatives)
    features[modalities.NEGATIVE_TEXT_LABELS] = tf.unique(
        new_negatives)[0][1:(new_num_negatives + 1)]

    # Negative integer labels are now obsolete:
    if modalities.NEGATIVE_LABELS in features:
      logging.info('Removing obsolete field %s from features.',
                   modalities.NEGATIVE_LABELS)
      features.pop(modalities.NEGATIVE_LABELS)

    return features


@dataclasses.dataclass
class AddRandomPrompts(NamedPreprocessOp):
  """Adds random promts to promptable text features.

  The op does the following:

  1. Get the set of unique label strings across all promptable modalities.
  2. Add a different random prompt to each unique string.
  3. Ensure that no prompts are added to unpromptable or empty strings.
  4. For each promptable modality, index back into the prompted label set to get
     final prompted label array.

  Attributes:
    promptable_modalities: Tuple of modality names that should be prompted.
  """

  promptable_modalities: Tuple[str, ...] = (modalities.INSTANCE_TEXT_LABELS,
                                            modalities.NEGATIVE_TEXT_LABELS)

  def apply(self, features: Features) -> Features:
    if image_ops.SEED_KEY not in features:
      raise ValueError('A random seed is required for prompt sampling.')

    rngs = tf.random.experimental.stateless_split(features[image_ops.SEED_KEY])
    features[image_ops.SEED_KEY] = rngs[0]
    op_seed = rngs[1]

    # Get set of labels and inverse indices:
    labels = [features[modality] for modality in self.promptable_modalities]
    unprompted_set, indices = tf.unique(tf.concat(labels, axis=0))
    index_list = tf.split(indices, [tf.shape(label)[0] for label in labels])

    # Add a random prompt to each label text in the set:
    random_templates = _sample_random_prompt_templates(
        num_samples=tf.shape(unprompted_set)[0], seed=op_seed)
    prompted_set = tf.map_fn(
        _add_prompt, (unprompted_set, random_templates),
        fn_output_signature=tf.TensorSpec([], tf.string))

    # Only apply prompts to promptable labels:
    is_promptable = tf.strings.regex_full_match(
        unprompted_set, f'[^{NOT_PROMPTABLE_MARKER}].*')
    prompted_set = tf.where(is_promptable, prompted_set, unprompted_set)

    # Do not apply prompts to empty strings (padding):
    prompted_set = tf.where(
        tf.equal(unprompted_set, PADDING_QUERY), PADDING_QUERY, prompted_set)

    # Replace text features with prompted versions:
    for modality, indices in zip(self.promptable_modalities, index_list):
      features[modality] = tf.gather(prompted_set, indices)

    # Add indicator that labels are now prompted:
    features['is_prompted'] = tf.constant(True)

    return features


@dataclasses.dataclass
class RemovePromptabilityMarker(NamedPreprocessOp):
  """Removes any promptability markers from text labels."""

  promptable_modalities: Tuple[str, ...] = (modalities.INSTANCE_TEXT_LABELS,
                                            modalities.NEGATIVE_TEXT_LABELS)

  def apply(self, features: Features) -> Features:

    # Remove non-promptable marker, if present:
    for key in self.promptable_modalities:
      features[key] = remove_promptability_marker(features[key])

    return features


@dataclasses.dataclass(frozen=True)
class SingleToMultiLabel(NamedPreprocessOp):
  """Converts instance labels to multi-label representation.

  Attributes:
    max_num_labels: Maximum number of per-instance labels.
    single_to_multi: Sequence of (src, tgt) tuples for modalities that need to
      be converted from single to multi-label representation.
  """

  max_num_labels: int = 100
  single_to_multi: Sequence[Tuple[str, str]] = (
      (modalities.INSTANCE_TEXT_LABELS, modalities.INSTANCE_TEXT_MULTI_LABELS),
      (modalities.INSTANCE_LABELS, modalities.INSTANCE_MULTI_LABELS))

  def apply(self, features: Features) -> Features:
    """Convert single label instances into multi-label."""

    features_new = dict(features)
    for src_name, tgt_name in self.single_to_multi:
      if src_name in features:
        src = features_new.pop(src_name)
        padding_value = transforms.get_padding_value(src.dtype)  # pytype: disable=attribute-error  # allow-recursive-types
        if padding_value is None:
          raise ValueError(f'Do not know how to pad {src.dtype} tensors.')  # pytype: disable=attribute-error  # allow-recursive-types

        tgt = tf.expand_dims(src, axis=-1)
        tgt = tf.pad(
            tgt,
            [(0, 0), (0, self.max_num_labels - 1)],
            constant_values=padding_value)
        tgt = tf.ensure_shape(tgt, [None, self.max_num_labels])
        features_new[tgt_name] = tgt

    return features_new


@dataclasses.dataclass
class AddQuerySet(NamedPreprocessOp):
  """Constructs a set of text queries from instance labels, for each example.

  This function will take positive and (optionally) negative text labels and
  apply "tf.unique" to create a set of queries. An empty query will be prepended
  to the query set account for padding.

  The queries define the per-example label space that is used for training.

  Attributes:
    max_queries: The maximum number of queries per example.
    include_negatives: Whether "negative" labels should be included in the query
      set. Negative labels are defined for datasets like LVIS or OpenImages. For
      referring expression datasets, no negatives exists, so nothing is added.
    lower: Whether to lower-case all queries. Must be set to True.
    instance_text_multi_labels_key: Key for the instance text labels.
    negative_text_labels_key: Key for the negative instance text labels.
    instance_multi_labels_key: For creating the label mapping.
    negative_labels_key: For creating the label mapping.
    text_queries_key: Key for the query set.
  """

  max_queries: int
  include_negatives: bool
  lower: bool = True

  instance_text_multi_labels_key: str = modalities.INSTANCE_TEXT_MULTI_LABELS
  negative_text_labels_key: str = modalities.NEGATIVE_TEXT_LABELS
  instance_multi_labels_key: str = modalities.INSTANCE_MULTI_LABELS
  negative_labels_key: str = modalities.NEGATIVE_LABELS
  text_queries_key: str = modalities.TEXT_QUERIES

  def __post_init__(self):
    if not self.lower:
      raise ValueError('The `lower` attribute exists only for backwards-'
                       'compatibility. It must be set to True.')

  def _get_unique_labelset(self, features):
    """Gets a unique set of queries and corresponding labels for one image."""
    instance_desc_orig = features[self.instance_text_multi_labels_key]
    instance_desc = tf.reshape(instance_desc_orig, (-1,))
    num_positive_labels = tf.shape(instance_desc)[0]
    if self.include_negatives:
      neg_desc = features[self.negative_text_labels_key]
      all_queries = tf.concat([instance_desc, neg_desc], axis=0)
    else:
      all_queries = instance_desc
    all_queries = _canonicalize_string_tf(all_queries)

    # Get unique labelset for this example and make sure that pad (empty text)
    # has index 0. This is done by prepending the empty string to the flattened
    # instance descriptions before applying tf.unique. The unique per example
    # labelset is then saved in features[modalities.TEXT_QUERIES].
    all_queries = tf.concat([tf.constant([PADDING_QUERY]), all_queries], 0)
    text_label_set, instance_labels = tf.unique(all_queries)
    instance_labels -= 1  # Shift labels so padding is -1.
    instance_labels = instance_labels[1:]  # Remove padding query label.
    instance_labels = instance_labels[:num_positive_labels]  # Remove negatives.

    # Crop or pad to len(max_queries):
    diff = self.max_queries - tf.shape(text_label_set)[0]
    queries = tf.cond(
        diff < 0, lambda: text_label_set[:self.max_queries], lambda: tf.pad(  # pylint: disable=g-long-lambda
            text_label_set, [(0, diff)], constant_values=PADDING_QUERY))
    instance_labels = tf.where(instance_labels < self.max_queries,
                               instance_labels, -1)
    instance_labels = tf.reshape(instance_labels, tf.shape(instance_desc_orig))

    return queries, instance_labels

  def apply(self, features: Features) -> Features:
    queries, instance_labels = self._get_unique_labelset(features)

    # Labels are now per-example indices into queries. Use -1 as padding.
    features[PER_EXAMPLE_INSTANCE_MULTI_LABELS] = instance_labels
    features[self.text_queries_key] = tf.ensure_shape(queries,
                                                      [self.max_queries])
    return features


@dataclasses.dataclass
class ClipTokenizeQueries(preprocess_spec.PreprocessOp):
  """Converts text query strings to integer tokens.

  Note that the features keys `queries` is used for both, text strings and
  token integers.

  Attributes:
    max_token_len: The maximum length of the queries after tokenization.
  """
  max_token_len: int
  text_queries_key: str = modalities.TEXT_QUERIES
  text_queries_tokenized_key: str = modalities.TEXT_QUERIES_TOKENIZED

  def __call__(self, features: image_ops.Features) -> image_ops.Features:
    if self.text_queries_key not in features:
      return features

    def _tokenize(inp):
      # Adapted from https://github.com/openai/CLIP/blob/main/clip/clip.py
      inp_shape = inp.shape
      inp = inp.reshape(-1)
      all_tokens = [
          clip_tokenizer.tokenize(text.decode('utf-8'), self.max_token_len)
          for text in inp
      ]
      result = np.zeros((len(all_tokens), self.max_token_len), dtype=np.int64)
      for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = np.asarray(tokens[:self.max_token_len])
      result = result.reshape(*inp_shape, self.max_token_len)
      return result

    tf_tokenize = functools.partial(tf.numpy_function, _tokenize,
                                    Tout=tf.int64)
    text_queries = features[self.text_queries_key]
    features[self.text_queries_tokenized_key] = tf.ensure_shape(
        tf_tokenize([text_queries]),
        text_queries.shape + [self.max_token_len])  # pytype: disable=attribute-error
    return features


def _multilabel_to_multihot(labels: tf.Tensor, num_classes: int) -> tf.Tensor:
  """Converts labels from multi-label to multi-hot representation.

  In the multi-label representation, labels have shape [...,
  max_num_labels_per_instance] and are integers in [0, num_classes], with -1 for
  padding.

  In the multi-hot representation, labels have shape [..., num_classes + 1] (+1
  due to padding) and are binary vectors with a 1 for each class that applies to
  that instance.

  Args:
    labels: [..., max_num_labels_per_instance] multi-labels.
    num_classes: Number of classes, including padding.

  Returns:
    [..., num_classes + 1] multi-hot label array.
  """

  # Labels are zero-indexed, with -1 for padding (TFDS convention). Add 1 such
  # that multi-hot index 0 means padding:
  labels = tf.one_hot(labels + 1, num_classes)
  labels = tf.reduce_max(labels, axis=-2)  # Combine up multi-labels.

  # Update padding label such that it is 1 iff there are no real labels. This
  # is necessary because multi-labels are padded, which means that the padding
  # label is initally hot for almost all real instances:
  is_padding = tf.cast(
      tf.reduce_all(labels[..., 1:] == 0, axis=-1, keepdims=True), labels.dtype)
  return tf.concat([is_padding, labels[..., 1:]], axis=-1)


@dataclasses.dataclass
class ConvertToScenic(NamedPreprocessOp):
  """Image processing op that converts features to Scenic format.

  Attributes:
    input_range: Tuple of minimum and maximum value to which the image will be
      scaled, e.g. (-1.0, 1.0). `None` defaults to the TensorFlow standard of
      (0.0, 1.0).
  """

  input_range: Optional[Tuple[float, float]]

  def apply(self, features: Features) -> Features:
    image = tf.image.convert_image_dtype(features['image'], dtype=tf.float32)

    if self.input_range is not None:
      image = image * (self.input_range[1] -
                       self.input_range[0]) + self.input_range[0]

    image_size = tf.cast(tf.shape(image)[:2], dtype=tf.int32)

    target = {
        'boxes':
            _convert_tf_boxes_to_xyxy(features[modalities.BOXES], image_size),
        'labels':
            _multilabel_to_multihot(
                labels=features[PER_EXAMPLE_INSTANCE_MULTI_LABELS],
                num_classes=tf.shape(features[modalities.TEXT_QUERIES])[0]),
    }

    out_features = {
        'inputs': image,
        'label': target,
        'queries': features[modalities.TEXT_QUERIES_TOKENIZED],
        'batch_mask': tf.ones((), dtype=tf.float32),
    }

    return detr_transforms.NormalizeBoxes()(out_features)


@dataclasses.dataclass
class StatelessLookupTable:
  """Basic lookup table which doesn't use stateful ops."""
  keys: tf.Tensor
  values: tf.Tensor
  default_value: Optional[tf.Tensor] = None

  def lookup(self, keys: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(keys)
    keys = tf.reshape(keys, [-1])
    values = tf.map_fn(self.lookup_single, keys,
                       dtype=self.values.dtype)
    return tf.reshape(values, shape)

  def lookup_single(self, key: tf.Tensor) -> tf.Tensor:
    """Return value corresponding to `key`."""
    key_string = tf.strings.as_string(key)
    labels_string = tf.strings.reduce_join(tf.strings.as_string(self.keys),
                                           separator=' | ')
    idx_equal = tf.squeeze(tf.where(self.keys == key))
    n = tf.size(idx_equal)
    if self.default_value is None:  # Must exist, no backup.
      msg = tf.strings.reduce_join(
          ['Could not find ', key_string, ' in label space:', labels_string])
      tf.assert_greater(n, 0, msg)
    msg = tf.strings.reduce_join([
        'Found ', tf.strings.as_string(n), ' matches for ', key_string,
        ' in label space:', labels_string])
    tf.assert_less(n, 2, msg)
    if self.default_value is None:
      return self.values[idx_equal]
    else:
      return tf.cond(n == 0, lambda: self.default_value,
                     lambda: self.values[idx_equal])
