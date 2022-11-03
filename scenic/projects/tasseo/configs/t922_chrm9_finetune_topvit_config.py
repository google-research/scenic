# pylint: disable=line-too-long
r"""Default configs for TopVit on t922_chrm9 baselines.

"""
# pylint: disable=line-too-long

from scenic.projects.tasseo.configs import abnormality_topvit_config as config_module


_ABNORMALITY = 't922_chrm9'
_SPLIT = 'finetune'


def get_config(*args, **kwargs):
  return config_module.get_config(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)


def get_hyper(*args, **kwargs):
  return config_module.get_hyper(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)
