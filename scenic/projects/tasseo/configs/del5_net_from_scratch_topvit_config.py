# pylint: disable=line-too-long
r"""Default configs for TopVit on del5_net baselines.

"""
# pylint: disable=line-too-long

from scenic.projects.tasseo.configs import abnormality_topvit_config as config_module


_ABNORMALITY = 'del5_net'
_SPLIT = 'start_from_scratch'


def get_config(*args, **kwargs):
  return config_module.get_config(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)


def get_hyper(*args, **kwargs):
  return config_module.get_hyper(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)
