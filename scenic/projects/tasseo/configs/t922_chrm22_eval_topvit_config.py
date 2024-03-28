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
r"""Default configs for TopVit on t922_chrm22 baselines.

"""
# pylint: disable=line-too-long

from scenic.projects.tasseo.configs import abnormality_topvit_config as config_module


_ABNORMALITY = 't922_chrm22'
_SPLIT = 'eval'


def get_config(*args, **kwargs):
  return config_module.get_config(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)


def get_hyper(*args, **kwargs):
  return config_module.get_hyper(
      *args, job_type=_SPLIT, abnormality=_ABNORMALITY, **kwargs)
