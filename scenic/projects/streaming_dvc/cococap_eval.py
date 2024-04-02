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

"""COCO caption evaluator, matching official implementations."""

from coco_caption import meteor
from coco_caption import spice
from coco_caption import upp_tokenizer
from pycocoevalcap.bleu import bleu
from pycocoevalcap.cider import cider
from pycocoevalcap.rouge import rouge

import numpy as np
import six
from six.moves import zip


class CustomMeteor(meteor.Meteor):
  """Meteor evaluator, consistent with official COCO implementation."""

  def compute_score(self, gts, res):
    """Compute METEOR scores."""
    with self.lock:
      assert sorted(gts.keys()) == sorted(res.keys())
      img_ids = sorted(gts.keys())

      eval_line = 'EVAL ||| '
      stats = self._stat(img_ids, res, gts)
      eval_line += ' ||| '.join(stats)
      # pytype: disable=attribute-error
      self.meteor_p.stdin.write(six.ensure_binary(eval_line + '\n'))
      self.meteor_p.stdin.flush()
      scores = [float(six.ensure_str(self.meteor_p.stdout.readline()))
                for _ in img_ids]
      # the aggregated value of the jar differs from the mean of other values
      score = self.meteor_p.stdout.readline()
      # pytype: enable=attribute-error
      # do not close the file inside this function to keep it open for full eval
    return float(score), np.asarray(scores)


class CustomCOCOEvalCap:
  """COCO caption evaluator that matches the external implementation."""

  def __init__(self, coco, coco_res, eval_meteor_spice=False):
    self.eval_imgs = []
    self.eval = {}
    self.img_to_eval = {}
    self.coco = coco
    self.coco_res = coco_res
    self.params = {'image_id': coco.getImgIds()}
    # meteor and spice evaluation needs additional data resources which can
    # not run with the default xm launcher. We provide the option to disable it.
    self.eval_meteor_spice = eval_meteor_spice

  def evaluate(self):
    """Run evaluation."""
    img_ids = self.params['image_id']
    gts = {}
    res = {}
    for img_id in img_ids:
      gts[img_id] = self.coco.imgToAnns[img_id]
      res[img_id] = self.coco_res.imgToAnns[img_id]

    # =================================================
    # Set up scorers
    # =================================================
    print('tokenization...')
    gts = upp_tokenizer.tokenize(gts)
    res = upp_tokenizer.tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print('setting up scorers...')
    scorers = [
        (rouge.Rouge(), 'ROUGE_L'),
        (cider.Cider(), 'CIDEr'),
        (bleu.Bleu(), 'BLEU-4'),
    ]
    if self.eval_meteor_spice:
      scorers.extend([
          (CustomMeteor(), 'Meteor'),
          (spice.Spice(), 'Spice'),
      ])

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
      print('computing %s score...' % (scorer.method()))
      score, scores = scorer.compute_score(gts, res)
      if isinstance(method, list):
        for sc, scs, m in zip(score, scores, method):
          self.setEval(sc, m)
          self.setImgToEvalImgs(scs, list(gts.keys()), m)
          print('%s: %0.3f' % (m, sc))
      else:
        if method == 'BLEU-4' and isinstance(score, list):
          score = score[-1]
        self.setEval(score, method)
        self.setImgToEvalImgs(scores, list(gts.keys()), method)
        print('%s: %0.3f' % (method, score))
    self.setEvalImgs()

  def setEval(self, score, method):  # pylint: disable=invalid-name
    self.eval[method] = score

  def setImgToEvalImgs(  # pylint: disable=invalid-name
      self, scores, img_ids, method):
    for img_id, score in zip(img_ids, scores):
      if img_id not in self.img_to_eval:
        self.img_to_eval[img_id] = {}
        self.img_to_eval[img_id]['image_id'] = img_id
      self.img_to_eval[img_id][method] = score

  def setEvalImgs(self):  # pylint: disable=invalid-name
    self.eval_imgs = [eval for _, eval in self.img_to_eval.items()]
