"""Common adaptation eval configuration."""

import ml_collections


def get_config():
  return ml_collections.ConfigDict()


def fixed(hyper, **kw):
  return hyper.zipit(
      [hyper.fixed(f'config.{k}', v, length=1) for k, v in kw.items()])


# pylint: disable=line-too-long
def imagenet(hyper,
             hres,
             lres,
             crop='random_crop',
             steps=20_000,
             warmup=500,
             mixup=None,
             randaug=False):
  """ImageNet task with standard labels and ReaL labels."""
  common = '|value_range(-1, 1)'
  common += '|onehot(1000, key="{lbl}", key_result="labels")'
  common += '|keep("image", "labels")'
  if crop == 'random_crop':
    pp_train = f'decode|resize({hres})|random_crop({lres})|flip_lr'
  elif crop == 'inception_crop':
    pp_train = f'decode_jpeg_and_inception_crop({lres})|flip_lr'
  else:
    raise ValueError(f'{crop} not in (random_crop, inception_crop).')
  if randaug:
    pp_train = pp_train + '|randaug(2, 15)'
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

  fixed_params = {
      'dataset_configs.dataset': 'imagenet2012',
      'dataset_configs.train_split': 'train[:99%]',
      'dataset_configs.pp_train': pp_train,
      'dataset_configs.val_split': [
          ('val', 'imagenet2012', 'train[99%:]', pp_val),
          ('test', 'imagenet2012', 'validation', pp_val),
          ('v2', 'imagenet_v2', 'test', pp_val),
          ('real', 'imagenet2012_real', 'validation', pp_real),
          ('y/val_resize', 'imagenet2012', 'train[99%:]', pp_val),
          ('y/test_resize', 'imagenet2012', 'validation', pp_val),
          ('y/v2_resize', 'imagenet_v2', 'test', pp_val),
          ('y/real_resize', 'imagenet2012_real', 'validation', pp_real),
          ('y/val_resize_crop', 'imagenet2012', 'train[99%:]',
           pp_val_resize_crop),
          ('y/test_resize_crop', 'imagenet2012', 'validation',
           pp_val_resize_crop),
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
      ],
      'dataset_configs.num_classes': 1000,
      'lr_configs.warmup_steps': warmup,
      'lr_configs.total_steps': steps,
      'lr_configs.steps_per_cycle': steps,
      'num_training_steps': steps,
  }
  if mixup is not None:
    fixed_params['mixup.p'] = mixup

  return fixed(hyper, **fixed_params)


def task(hyper,
         name,
         train,
         val,
         n_cls,
         hres,
         lres,
         crop,
         steps,
         warmup,
         test='test',
         base_pp='',
         randaug=False):
  """Vision task with val and test splits."""
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
  if randaug:
    pp_train = (f'decode_jpeg_and_inception_crop({lres})|flip_lr'
                '|randaug(2, 15)'
                '|value_range(-1, 1)'
                '|onehot({1000}, key="label", key_result="labels")'
                '|keep("image", "labels")')
  pp_eval = f'decode|{base_pp}resize({lres})' + common
  pp_eval_resize_crop = f'decode|{base_pp}resize({hres})|central_crop({lres})' + common
  pp_eval_resmall_crop = f'decode|{base_pp}resize_small({hres})|central_crop({lres})' + common

  return fixed(
      hyper, **{
          'dataset_configs.dataset': name,
          'dataset_configs.train_split': train,
          'dataset_configs.pp_train': pp_train,
          'dataset_configs.val_split': [
              ('val', name, val, pp_eval),
              ('y/val_resize', name, val, pp_eval),
              ('y/val_resize_crop', name, val, pp_eval_resize_crop),
              ('y/val_resmall_crop', name, val, pp_eval_resmall_crop),
              ('test', name, test, pp_eval),
              ('y/test_resize', name, test, pp_eval),
              ('y/test_resize_crop', name, test, pp_eval_resize_crop),
              ('y/test_resmall_crop', name, test, pp_eval_resmall_crop),
          ],
          'dataset_configs.num_classes': n_cls,
          'lr_configs.warmup_steps': warmup,
          'lr_configs.total_steps': steps,
          'lr_configs.steps_per_cycle': steps,
          'num_training_steps': steps,
      })


# pylint: enable=line-too-long
