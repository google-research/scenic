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
r"""Default configs for PolyVit.

"""

import ml_collections

IMAGE_TASKS = ['imagenet', 'cifar10', 'cifar100', 'oxford_iiit_pet', 'resisc45']
VIDEO_TASKS = ['kinetics400', 'moments_in_time']
AUDIO_TASKS = ['vggsound', 'audioset']


# The size of the these datasets changes as videos are removed from YouTube.
# Set this appropriately.
KINETICS_400_TRAIN_SIZE = 214834
KINETICS_400_VAL_SIZE = 17637
KINETICS_400_TEST_SIZE = 34579
MIT_TRAIN_SIZE = 791297
AUDIOSET_TRAIN_SIZE = 20361
VGGSOUND_TRAIN_SIZE = 172427


def imagenet(config,
             hres,
             lres,
             lr,
             crop='random_crop',
             steps=20_000,
             warmup=500,
             mixup=None):
  """Vision task with val and test splits."""
  del warmup
  common = '|value_range(-1, 1)'
  common += '|onehot(1000, key="{lbl}", key_result="labels")'
  common += '|keep("image", "labels")'
  if crop == 'random_crop':
    pp_train = f'decode|resize({hres})|random_crop({lres})|flip_lr'
  elif crop == 'inception_crop':
    pp_train = f'decode_jpeg_and_inception_crop({lres})|flip_lr'
  else:
    raise ValueError(f'{crop} not in (random_crop, inception_crop).')
  pp_train += common.format(lbl='label')
  pp_val = f'decode|resize({lres})' + common.format(lbl='label')
  pp_real = f'decode|resize({lres})' + common.format(lbl='real_label')
  pp_val_resize_crop = f'decode|resize({hres})|central_crop({lres})' + common.format(
      lbl='label')
  pp_real_resize_crop = f'decode|resize({hres})|central_crop({lres})' + common.format(
      lbl='real_label')
  pp_val_resmall_crop = f'decode|resize_small({hres})|central_crop({lres})' + common.format(
      lbl='label')
  pp_real_resmall_crop = f'decode|resize_small({hres})|central_crop({lres})' + common.format(
      lbl='real_label')

  config.datasets.bit_imagenet2012 = ml_collections.ConfigDict()
  config.datasets.bit_imagenet2012.dataset = 'imagenet2012'
  config.datasets.bit_imagenet2012.dataset_dir = None
  config.datasets.bit_imagenet2012.task = 'multilabel'
  config.datasets.bit_imagenet2012.data_dtype_str = 'float32'
  config.datasets.bit_imagenet2012.train_split = 'train[:99%]'
  config.datasets.bit_imagenet2012.val_split = [
      ('val', 'imagenet2012', 'train[99%:]', pp_val),
      ('test', 'imagenet2012', 'validation', pp_val),
      ('v2', 'imagenet_v2', 'test', pp_val),
      ('real', 'imagenet2012_real', 'validation', pp_real),
      ('y/val_resize', 'imagenet2012', 'train[99%:]', pp_val),
      ('y/test_resize', 'imagenet2012', 'validation', pp_val),
      ('y/v2_resize', 'imagenet_v2', 'test', pp_val),
      ('y/real_resize', 'imagenet2012_real', 'validation', pp_real),
      ('y/val_resize_crop', 'imagenet2012', 'train[99%:]', pp_val_resize_crop),
      ('y/test_resize_crop', 'imagenet2012', 'validation', pp_val_resize_crop),
      ('y/v2_resize_crop', 'imagenet_v2', 'test', pp_val_resize_crop),
      ('y/real_resize_crop', 'imagenet2012_real', 'validation',
       pp_real_resize_crop),
      ('y/val_resmall_crop', 'imagenet2012', 'train[99%:]',
       pp_val_resmall_crop),
      ('y/test_resmall_crop', 'imagenet2012', 'validation',
       pp_val_resmall_crop),
      ('y/v2_resmall_crop', 'imagenet_v2', 'test', pp_val_resmall_crop),
      ('y/real_resmall_crop', 'imagenet2012_real', 'validation',
       pp_real_resmall_crop),
  ]
  config.datasets.bit_imagenet2012.num_classes = 1000
  config.datasets.bit_imagenet2012.pp_train = pp_train
  config.datasets.bit_imagenet2012.pp_eval = ''
  config.datasets.bit_imagenet2012.prefetch_to_device = 2
  config.datasets.bit_imagenet2012.shuffle_buffer_size = 50_000

  config.model.heads.label.bit_imagenet2012 = ml_collections.ConfigDict()
  config.model.heads.label.bit_imagenet2012.hid_sizes = ()
  config.model.heads.label.bit_imagenet2012.classifier = 'token'

  config.lr_coefs.bit_imagenet2012 = lr
  config.batch_sampling_strategy_steps.bit_imagenet2012 = steps

  if mixup is not None:
    config.mixup.p = mixup


def task(config,
         dataset_id,
         name,
         train,
         val,
         lr,
         n_cls,
         hres,
         lres,
         crop,
         steps,
         warmup,
         test='test',
         base_pp=''):
  """Vision task with val and test splits."""
  del warmup
  common = '|value_range(-1, 1)'
  common += f'|onehot({n_cls}, key="label", key_result="labels")'
  common += '|keep("image", "labels")'
  if crop == 'random_crop':
    pp_train = f'decode|{base_pp}resize({hres})|random_crop({lres})|flip_lr'
  elif crop == 'inception_crop':
    pp_train = f'decode|{base_pp}inception_crop({lres})|flip_lr'
  elif not crop:
    pp_train = f'decode|{base_pp}resize({lres})|flip_lr'
  else:
    raise ValueError(f'{crop} not in ("random_crop", "inception_crop", "").')
  pp_train += common
  pp_eval = f'decode|{base_pp}resize({lres})' + common
  pp_eval_resize_crop = f'decode|{base_pp}resize({hres})|central_crop({lres})' + common
  pp_eval_resmall_crop = f'decode|{base_pp}resize_small({hres})|central_crop({lres})' + common

  config.datasets[dataset_id] = ml_collections.ConfigDict()
  config.datasets[dataset_id].dataset = name
  config.datasets[dataset_id].dataset_dir = None
  config.datasets[dataset_id].task = 'multilabel'
  config.datasets[dataset_id].data_dtype_str = 'float32'
  config.datasets[dataset_id].train_split = train
  config.datasets[dataset_id].val_split = [
      ('val', name, val, pp_eval),
      ('y/val_resize', name, val, pp_eval),
      ('y/val_resize_crop', name, val, pp_eval_resize_crop),
      ('y/val_resmall_crop', name, val, pp_eval_resmall_crop),
      ('test', name, test, pp_eval),
      ('y/test_resize', name, test, pp_eval),
      ('y/test_resize_crop', name, test, pp_eval_resize_crop),
      ('y/test_resmall_crop', name, test, pp_eval_resmall_crop),
  ]
  config.datasets[dataset_id].num_classes = n_cls
  config.datasets[dataset_id].pp_train = pp_train
  config.datasets[dataset_id].pp_eval = ''
  config.datasets[dataset_id].prefetch_to_device = 2
  config.datasets[dataset_id].shuffle_buffer_size = 50_000

  config.model.heads.label[dataset_id] = ml_collections.ConfigDict()
  config.model.heads.label[dataset_id].hid_sizes = ()
  config.model.heads.label[dataset_id].classifier = 'token'

  config.lr_coefs[dataset_id] = lr
  config.batch_sampling_strategy_steps[dataset_id] = steps


def get_config():
  """Returns the ViT experiment configuration for JFT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'vtab-polyvit'

  config.datasets = ml_collections.ConfigDict()
  config.model = ml_collections.ConfigDict()
  config.model.heads = ml_collections.ConfigDict()
  config.model.heads.label = ml_collections.ConfigDict()
  config.lr_coefs = ml_collections.ConfigDict()
  config.batch_sampling_strategy_steps = ml_collections.ConfigDict()

  if 'imagenet' in IMAGE_TASKS:
    imagenet(
        config,
        hres=448,
        lres=384,
        lr=0.03,
        crop='inception_crop',
        steps=20_000,
        warmup=500)
  if 'cifar100' in IMAGE_TASKS:
    task(
        config,
        'bit_cifar100',
        'cifar100',
        'train[:98%]',
        'train[98%:]',
        lr=0.03,
        n_cls=100,
        hres=448,
        lres=384,
        steps=10_000,
        warmup=500,
        crop='inception_crop')
  if 'cifar10' in IMAGE_TASKS:
    task(
        config,
        'bit_cifar10',
        'cifar10',
        'train[:98%]',
        'train[98%:]',
        lr=0.03,
        n_cls=10,
        hres=448,
        lres=384,
        steps=10_000,
        warmup=500,
        crop='inception_crop')
  if 'oxford_iiit_pet' in IMAGE_TASKS:
    task(
        config,
        'bit_oxford_iiit_pet',
        'oxford_iiit_pet',
        'train[:90%]',
        'train[90%:]',
        lr=0.03,
        n_cls=37,
        hres=448,
        lres=384,
        steps=500,
        warmup=100,
        crop='inception_crop')
  if 'resisc45' in IMAGE_TASKS:
    task(
        config,
        'bit_resisc45',
        'resisc45',
        'train[:60%]',
        'train[60%:80%]',
        lr=0.1,
        n_cls=45,
        hres=448,
        lres=384,
        steps=2500,
        warmup=200,
        crop='inception_crop',
        test='train[80%:]',
    )
  if 'kinetics400' in VIDEO_TASKS:
    config.datasets.kinetics400 = ml_collections.ConfigDict()
    config.datasets.kinetics400.task = 'label'
    config.datasets.kinetics400.modality = 'video'
    config.datasets.kinetics400.data_dtype_str = 'float32'
    # This is going to sample 32 frames, sampled at a stride of 2 from video.
    # kinetics400 videos has 250 frames.
    # And then it will uniformly take n_sampled_frames from there.
    # Maybe think more about this.
    # 32 stride 2 is the default of SlowFast.
    config.datasets.kinetics400.num_frames = 32
    config.datasets.kinetics400.stride = 2
    config.datasets.kinetics400.min_resize = 256
    config.datasets.kinetics400.crop_size = 224
    config.datasets.kinetics400.one_hot_labels = True
    config.datasets.kinetics400.zero_centering = True
    # Multicrop eval settings
    config.datasets.kinetics400.do_multicrop_test = True  # Do during training.
    config.datasets.kinetics400.log_test_epochs = 5
    # The effective batch size per host when testing is num_test_clips * test_batch_size  # pylint: disable=line-too-long
    config.datasets.kinetics400.num_test_clips = 8
    config.datasets.kinetics400.test_batch_size = 8  # Needs to be num_local_devices
    config.multicrop_clips_per_device = 2
    # Leaving this empty means that a full test is done each time.
    # About 4200 / 4 = 1050 steps on a 4-host setting (ie 4x4 TPU)
    # config.steps_per_test = 1000  # Number of test steps taken by each host.
    config.datasets.kinetics400.augmentation_params = ml_collections.ConfigDict(
    )
    config.datasets.kinetics400.augmentation_params.do_jitter_scale = True
    config.datasets.kinetics400.augmentation_params.scale_min_factor = 0.9
    config.datasets.kinetics400.augmentation_params.scale_max_factor = 1.33
    config.datasets.kinetics400.augmentation_params.prob_scale_jitter = 1.0
    config.datasets.kinetics400.augmentation_params.do_color_augment = True
    config.datasets.kinetics400.augmentation_params.prob_color_augment = 0.8
    config.datasets.kinetics400.augmentation_params.prob_color_drop = 0.1
    config.datasets.kinetics400.prefetch_to_device = 2
    config.datasets.kinetics400.base_dir = (
        '/path/to/dataset')
    config.datasets.kinetics400.tables = {
        'train': 'train.tfrecord@1024',
        'validation': 'validation.tfrecord@1024',
        'test': 'test.tfrecord@1024'
    }
    config.datasets.kinetics400.examples_per_subset = {
        'train': KINETICS_400_TRAIN_SIZE,
        'validation': KINETICS_400_VAL_SIZE,
        'test': KINETICS_400_TEST_SIZE
    }
    config.datasets.kinetics400.num_classes = 400

  if 'moments_in_time' in VIDEO_TASKS:
    config.datasets.moments_in_time = ml_collections.ConfigDict()
    config.datasets.moments_in_time.task = 'label'
    config.datasets.moments_in_time.modality = 'video'
    config.datasets.moments_in_time.data_dtype_str = 'float32'
    # This is going to sample 32 frames, sampled at a stride of 1 from video.
    # And then it will uniformly take n_sampled_frames from there.
    config.datasets.moments_in_time.num_frames = 32
    config.datasets.moments_in_time.stride = 2
    config.datasets.moments_in_time.min_resize = 256
    config.datasets.moments_in_time.crop_size = 224
    config.datasets.moments_in_time.one_hot_labels = True
    config.datasets.moments_in_time.zero_centering = True
    # Multicrop eval settings
    config.datasets.moments_in_time.do_multicrop_test = True
    config.datasets.moments_in_time.log_test_epochs = 2
    # The effective batch size per host when testing is num_test_clips * test_batch_size  # pylint: disable=line-too-long
    config.datasets.moments_in_time.num_test_clips = 8
    config.datasets.moments_in_time.test_batch_size = 8  # Needs to be num_local_devices
    # Leaving this empty means that a full test is done each time.
    # config.steps_per_test = 1000  # Number of test steps taken by each host.
    config.datasets.moments_in_time.augmentation_params = ml_collections.ConfigDict(
    )
    config.datasets.moments_in_time.augmentation_params.do_jitter_scale = True
    config.datasets.moments_in_time.augmentation_params.scale_min_factor = 0.9
    config.datasets.moments_in_time.augmentation_params.scale_max_factor = 1.33
    config.datasets.moments_in_time.augmentation_params.prob_scale_jitter = 1.0
    config.datasets.moments_in_time.augmentation_params.do_color_augment = True
    config.datasets.moments_in_time.augmentation_params.prob_color_augment = 0.8
    config.datasets.moments_in_time.augmentation_params.prob_color_drop = 0.1
    config.datasets.moments_in_time.augmentation_params.do_mixup = False
    config.datasets.moments_in_time.augmentation_params.mixup_alpha = 0.0
    config.datasets.moments_in_time.prefetch_to_device = 2
    config.datasets.moments_in_time.base_dir = (
        '/path/to/dataset')
    config.datasets.moments_in_time.tables = {
        'train': 'train.tfrecord@1024',
        'validation': 'validation.tfrecord@1024',
        'test': 'test.tfrecord@1024'
    }
    config.datasets.moments_in_time.examples_per_subset = {
        'train': MIT_TRAIN_SIZE,
        'validation': 33900,
        'test': 33900
    }
    config.datasets.moments_in_time.num_classes = 339

  if 'audioset' in AUDIO_TASKS:
    config.datasets.balanced_audioset = ml_collections.ConfigDict()
    audioset_config = config.datasets.balanced_audioset
    audioset_config.task = 'multilabel'
    audioset_config.modality = 'audio'
    audioset_config.data_dtype_str = 'float32'
    # List of modalities to load, supports `rgb`, `spectrogram` and `waveform`.
    # Note that it only specifies which modalities to load, not which to use,
    # which is controlled by config.model.modality_fusion
    audioset_config.modalities = ('spectrogram',)
    audioset_config.return_as_dict = True
    # This is going to sample 32 frames, sampled at a stride of 2 from video.
    # AudioSet videos has 250 frames.
    # 32 stride 2 is also the default of SlowFast.
    audioset_config.num_frames = 32
    audioset_config.stride = 2
    audioset_config.num_spec_frames = 8
    audioset_config.spec_stride = 1
    audioset_config.min_resize = 256
    audioset_config.crop_size = 224
    audioset_config.spec_shape = (100, 128)
    # 16000 samples per second.
    audioset_config.num_waveform_samples = 32256  # 6 * 7 * 768
    audioset_config.waveform_stride = 1
    audioset_config.one_hot_labels = True
    audioset_config.zero_centering = True
    # Class prior settings.
    audioset_config.class_weight_power = 0.3
    # Multicrop eval settings
    audioset_config.do_multicrop_test = True  # Do during training.
    audioset_config.log_test_epochs = 4
    # The effective batch size per host when testing is
    # num_test_clips * test_batch_size
    audioset_config.num_test_clips = 4
    audioset_config.test_batch_size = 8  # Needs to be num_local_devices
    config.multicrop_clips_per_device = 2
    # Leaving this empty means that a full test is done each time.
    # About 4200 / 4 = 1050 steps on a 4-host setting (ie 4x4 TPU)
    # config.steps_per_test = 1000  # Number of test steps taken by each host.
    audioset_config.augmentation_params = ml_collections.ConfigDict()
    audioset_config.augmentation_params.do_jitter_scale = True
    audioset_config.augmentation_params.scale_min_factor = 0.9
    audioset_config.augmentation_params.scale_max_factor = 1.33
    audioset_config.augmentation_params.prob_scale_jitter = 1.0
    audioset_config.augmentation_params.do_color_augment = True
    audioset_config.augmentation_params.prob_color_augment = 0.8
    audioset_config.augmentation_params.prob_color_drop = 0.1
    audioset_config.prefetch_to_device = 2
    # SpecAugment hyperparameters
    audioset_config.spec_augment = True
    audioset_config.spec_augment_params = ml_collections.ConfigDict()
    audioset_config.spec_augment_params.freq_mask_max_bins = 48
    audioset_config.spec_augment_params.freq_mask_count = 1
    audioset_config.spec_augment_params.time_mask_max_frames = 48
    audioset_config.spec_augment_params.time_mask_count = 4
    audioset_config.spec_augment_params.time_warp_max_frames = 1.0
    audioset_config.spec_augment_params.time_warp_max_ratio = 0
    audioset_config.spec_augment_params.time_mask_max_ratio = 0
    audioset_config.base_dir = (
        '/path/to/dataset')
    audioset_config.tables = {
        'train': 'balanced_train.se.melspec.tfrecord.sst@1024',
        'validation': 'eval.se.melspec.tfrecord.sst@1024',
        'test': 'eval.se.melspec.tfrecord.sst@1024',
    }
    audioset_config.examples_per_subset = {
        'train': 20361,
        'validation': 18589,
        'test': 18589
    }
    audioset_config.num_classes = 527

  if 'vggsound' in AUDIO_TASKS:
    config.datasets.vggsound = ml_collections.ConfigDict()
    vggsound_config = config.datasets.vggsound
    vggsound_config.task = 'label'
    vggsound_config.modality = 'audio'
    vggsound_config.data_dtype_str = 'float32'
    # List of modalities to load, supports `rgb`, `spectrogram` and `waveform`.
    # Note that it only specifies which modalities to load, not which to use,
    # which is controlled by config.model.modality_fusion
    vggsound_config.modalities = ('spectrogram',)
    vggsound_config.return_as_dict = True
    # This is going to sample 32 frames, sampled at a stride of 2 from video.
    # AudioSet videos has 250 frames.
    # 32 stride 2 is also the default of SlowFast.
    vggsound_config.num_frames = 32
    vggsound_config.stride = 2
    vggsound_config.num_spec_frames = 8
    vggsound_config.spec_stride = 1
    vggsound_config.min_resize = 256
    vggsound_config.crop_size = 224
    vggsound_config.spec_shape = (100, 128)
    # 16000 samples per second.
    vggsound_config.num_waveform_samples = 32256  # 6 * 7 * 768
    vggsound_config.waveform_stride = 1
    vggsound_config.one_hot_labels = True
    vggsound_config.zero_centering = True
    # Multicrop eval settings
    vggsound_config.do_multicrop_test = True  # Do during training.
    vggsound_config.log_test_epochs = 0.5
    # The effective batch size per host when testing is
    # num_test_clips * test_batch_size
    vggsound_config.num_test_clips = 4
    vggsound_config.test_batch_size = 8  # Needs to be num_local_devices
    config.multicrop_clips_per_device = 2
    # Leaving this empty means that a full test is done each time.
    # About 4200 / 4 = 1050 steps on a 4-host setting (ie 4x4 TPU)
    # config.steps_per_test = 1000  # Number of test steps taken by each host.
    vggsound_config.augmentation_params = ml_collections.ConfigDict()
    vggsound_config.augmentation_params.do_jitter_scale = True
    vggsound_config.augmentation_params.scale_min_factor = 0.9
    vggsound_config.augmentation_params.scale_max_factor = 1.33
    vggsound_config.augmentation_params.prob_scale_jitter = 1.0
    vggsound_config.augmentation_params.do_color_augment = True
    vggsound_config.augmentation_params.prob_color_augment = 0.8
    vggsound_config.augmentation_params.prob_color_drop = 0.1
    vggsound_config.prefetch_to_device = 2
    # SpecAugment hyperparameters
    vggsound_config.spec_augment = True
    vggsound_config.spec_augment_params = ml_collections.ConfigDict()
    vggsound_config.spec_augment_params.freq_mask_max_bins = 48
    vggsound_config.spec_augment_params.freq_mask_count = 1
    vggsound_config.spec_augment_params.time_mask_max_frames = 48
    vggsound_config.spec_augment_params.time_mask_count = 4
    vggsound_config.spec_augment_params.time_warp_max_frames = 1.0
    vggsound_config.spec_augment_params.time_warp_max_ratio = 0
    vggsound_config.spec_augment_params.time_mask_max_ratio = 0
    vggsound_config.base_dir = (
        '/path/to/dataset')
    vggsound_config.tables = {
        'train': 'train.rgb.25fps.wav.mel.spec.labels.sst@1024',
        'validation': 'test.rgb.25fps.wav.mel.spec.labels.sst@1024',
        'test.rgb.25fps.wav.mel.spec.labels.sst@1024',
    }
    vggsound_config.examples_per_subset = {
        'train': 172427,
        'validation': 14448,
        'test': 14448
    }
    vggsound_config.num_classes = 309

  # Model.
  config.model_name = 'polyvit'
  config.model.modalities = ml_collections.ConfigDict()
  config.model.modalities.num_layers = 0
  config.model.modalities.hidden_size = 768
  config.model.modalities.image = ml_collections.ConfigDict()
  config.model.modalities.image.patches = ml_collections.ConfigDict()
  config.model.modalities.image.patches.size = [16, 16]
  config.model.modalities.audio = ml_collections.ConfigDict()
  config.model.modalities.audio.patches = ml_collections.ConfigDict()
  config.model.modalities.audio.patches.size = [16, 16]
  config.model.modalities.audio.spec_shape = (100, 128)
  config.model.modalities.audio.num_spec_frames = 8
  config.model.modalities.video = ml_collections.ConfigDict()
  config.model.modalities.video.patches = ml_collections.ConfigDict()
  config.model.modalities.video.patches.size = [16, 16, 4]
  config.model.modalities.video.kernel_init_method = 'central_frame_initializer'

  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.num_heads = 12
  config.model.encoder.mlp_dim = 3072
  config.model.encoder.num_layers = 12
  config.model.encoder.representation_size = None
  config.model.encoder.attention_dropout_rate = 0.
  config.model.encoder.dropout_rate = 0.
  if 'kinetics400' in VIDEO_TASKS:
    config.model.heads.label.kinetics400 = ml_collections.ConfigDict()
    config.model.heads.label.kinetics400.hid_sizes = ()
    config.model.heads.label.kinetics400.output_proj_zero_init = True
    config.model.heads.label.kinetics400.classifier = 'token'
  if 'moments_in_time' in VIDEO_TASKS:
    config.model.heads.label.moments_in_time = ml_collections.ConfigDict()
    config.model.heads.label.moments_in_time.hid_sizes = ()
    config.model.heads.label.moments_in_time.output_proj_zero_init = True
    config.model.heads.label.moments_in_time.classifier = 'token'
  if 'vggsound' in AUDIO_TASKS:
    config.model.heads.label.vggsound = ml_collections.ConfigDict()
    config.model.heads.label.vggsound.hid_sizes = ()
    config.model.heads.label.vggsound.output_proj_zero_init = True
    config.model.heads.label.vggsound.classifier = 'token'
  if 'audioset' in AUDIO_TASKS:
    config.model.heads.label.balanced_audioset = ml_collections.ConfigDict()
    config.model.heads.label.balanced_audioset.hid_sizes = ()
    config.model.heads.label.balanced_audioset.output_proj_zero_init = True
    config.model.heads.label.balanced_audioset.classifier = 'token'
  config.model_dtype_str = 'float32'

  # Training.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scenic.momentum_hp'
  optim.weight_decay = 0.0
  config.optimizer = optim
  config.l2_decay_factor = 0
  # We customize the gradient clipping depending on the dataset.
  optim.max_grad_norm = None
  config.max_grad_norm = 1.0

  config.num_training_epochs = None
  config.batch_sizes = ml_collections.ConfigDict()
  if 'imagenet' in IMAGE_TASKS:
    config.batch_sizes.bit_imagenet2012 = 512
  if 'cifar10' in IMAGE_TASKS:
    config.batch_sizes.bit_cifar10 = 512
  if 'cifar100' in IMAGE_TASKS:
    config.batch_sizes.bit_cifar100 = 512
  if 'oxford_iiit_pet' in IMAGE_TASKS:
    config.batch_sizes.bit_oxford_iiit_pet = 512
  if 'resisc45' in IMAGE_TASKS:
    config.batch_sizes.bit_resisc45 = 512
  if 'vggsound' in AUDIO_TASKS:
    config.batch_sizes.vggsound = 64
  if 'audioset' in AUDIO_TASKS:
    config.batch_sizes.balanced_audioset = 64
  if 'kinetics400' in VIDEO_TASKS:
    config.batch_sizes.kinetics400 = 64
  if 'moments_in_time' in VIDEO_TASKS:
    config.batch_sizes.moments_in_time = 64

  config.num_training_steps = 0
  if 'imagenet' in IMAGE_TASKS:
    config.num_training_steps += 20_000
  if 'cifar10' in IMAGE_TASKS:
    config.num_training_steps += 10_000
  if 'cifar100' in IMAGE_TASKS:
    config.num_training_steps += 10_000
  if 'oxford_iiit_pet' in IMAGE_TASKS:
    config.num_training_steps += 500
  if 'resisc45' in IMAGE_TASKS:
    config.num_training_steps += 2500
  if 'vggsound' in AUDIO_TASKS:
    vggsound_steps_per_epoch = VGGSOUND_TRAIN_SIZE // config.batch_sizes.vggsound
    vggsound_n_epochs = 50
    vggsound_steps = vggsound_steps_per_epoch * vggsound_n_epochs
    config.num_training_steps += vggsound_steps
  if 'audioset' in AUDIO_TASKS:
    audioset_steps_per_epoch = AUDIOSET_TRAIN_SIZE // config.batch_sizes.balanced_audioset
    audioset_n_epochs = 50
    audioset_steps = audioset_steps_per_epoch * audioset_n_epochs
    config.num_training_steps += audioset_steps
  if 'kinetics400' in VIDEO_TASKS:
    kinetics_steps_per_epoch = KINETICS_400_TRAIN_SIZE // config.batch_sizes.kinetics400
    kinetics_n_epochs = 30
    kinetics_steps = kinetics_steps_per_epoch * kinetics_n_epochs
    config.num_training_steps += kinetics_steps
  if 'moments_in_time' in VIDEO_TASKS:
    mit_steps_per_epoch = MIT_TRAIN_SIZE // config.batch_sizes.moments_in_time
    mit_n_epochs = 10
    mit_steps = mit_steps_per_epoch * mit_n_epochs
    config.num_training_steps += mit_steps

  config.log_eval_steps = 5000
  config.rng_seed = 42

  config.stochastic_droplayer_rates = ml_collections.ConfigDict()
  config.stochastic_droplayer_rates.vggsound = 0.3
  config.stochastic_droplayer_rates.balanced_audioset = 0.3
  config.mixups = ml_collections.ConfigDict()
  if 'vggsound' in AUDIO_TASKS:
    config.mixups.vggsound = ml_collections.ConfigDict()
    config.mixups.vggsound.mixmod = True
    config.mixups.vggsound.alpha = 0.3
  if 'audioset' in AUDIO_TASKS:
    config.mixups.balanced_audioset = ml_collections.ConfigDict()
    config.mixups.balanced_audioset.mixmod = True
    config.mixups.balanced_audioset.alpha = 0.3

  config.init_from = ml_collections.ConfigDict()
  config.init_from.init_from_vit = True
  config.init_from.model_config = None
  # Download pretrained ImageNet checkpoints from here:
  # https://github.com/google-research/scenic/tree/main/scenic/projects/baselines (checkpoint_format = 'scenic')  pylint: disable=line-too-long
  # https://github.com/google-research/vision_transformer (checkpoint_format = 'big_vision')  pylint: disable=line-too-long
  config.init_from.checkpoint_path = 'path_to_checkpoint_of_vit_b_16'
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  # Only used for video heads, "resize" for image heads.
  config.init_from.positional_embed_size_change = 'tile'

  # Learning rate.
  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  sched.lr_configs.total_steps = config.num_training_steps
  sched.lr_configs.steps_per_cycle = sched.lr_configs.total_steps
  sched.lr_configs.warmup_steps = 0.0
  sched.lr_configs.base_learning_rate = 1.0

  if len(IMAGE_TASKS) > 1:
    sched.lr_configs.warmup_steps += 500
  if 'vggsound' in AUDIO_TASKS:
    sched.lr_configs.warmup_steps += vggsound_steps_per_epoch * 2.5
    config.lr_coefs.vggsound = 0.5
  if 'audioset' in AUDIO_TASKS:
    sched.lr_configs.warmup_steps += audioset_steps_per_epoch * 2.5
    config.lr_coefs.balanced_audioset = 0.5
  if 'kinetics400' in VIDEO_TASKS:
    sched.lr_configs.warmup_steps += kinetics_steps_per_epoch * 2.5
    config.lr_coefs.kinetics400 = 0.1
  if 'moments_in_time' in VIDEO_TASKS:
    sched.lr_configs.warmup_steps += mit_steps_per_epoch * 2.5
    config.lr_coefs.moments_in_time = 0.25
  config.schedule = ml_collections.ConfigDict({'all': sched})

  if 'vggsound' in AUDIO_TASKS:
    config.batch_sampling_strategy_steps.vggsound = vggsound_steps
  if 'audioset' in AUDIO_TASKS:
    config.batch_sampling_strategy_steps.balanced_audioset = audioset_steps
  if 'kinetics400' in VIDEO_TASKS:
    config.batch_sampling_strategy_steps.kinetics400 = kinetics_steps
  if 'moments_in_time' in VIDEO_TASKS:
    config.batch_sampling_strategy_steps.moments_in_time = mit_steps

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.log_summary_steps = 100
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.trial = 0
  return config


# pylint: enable=line-too-long
