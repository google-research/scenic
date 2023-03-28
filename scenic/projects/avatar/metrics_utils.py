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
