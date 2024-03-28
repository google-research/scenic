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

"""Calculation of metrics to evaluate tokgen generation performance."""

from jiwer.measures import compute_measures


def word_error_rate(ref, hyp):
  ref = [format_string(s) for s in ref]
  hyp = [format_string(s) for s in hyp]
  scores = compute_measures(ref, hyp)
  wer = scores['wer']
  cor_c, sub_c = scores['hits'], scores['substitutions']
  del_c, ins_c = scores['deletions'], scores['insertions']
  total_c = del_c + sub_c + cor_c
  rates = (del_c / total_c, ins_c / total_c, sub_c / total_c, cor_c / total_c)
  return wer, rates


def format_string(s):
  # Replaces multiple spaces by a single space
  s = ' '.join(s.split())
  return s
