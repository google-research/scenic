# pylint: disable=line-too-long
r"""Default configs for TopVit on del5_simple fine tuned baselines.

"""
# pylint: disable=line-too-long

from scenic.projects.tasseo.configs import abnormality_topvit_config as config_module


_ABNORMALITY = 'del5_simple'
_SPLIT = 'eval'


def get_config(*args, **kwargs):
  return config_module.get_config(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)


def get_hyper(*args, **kwargs):
  return config_module.get_hyper(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)
