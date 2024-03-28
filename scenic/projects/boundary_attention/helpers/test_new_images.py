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

"""Code to test Boundary Attention on new images."""

import pickle

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import PIL
from scenic.projects.boundary_attention.configs import base_config
from scenic.projects.boundary_attention.helpers import train_utils
from scenic.projects.boundary_attention.helpers import viz_utils
from tensorflow.io import gfile


flags.DEFINE_integer('height', 216, 'Height of input.')
flags.DEFINE_integer('width', 216, 'Width of input.')
flags.DEFINE_integer('rng_seed', 0, 'Rng seed.')
flags.DEFINE_string('weights_dir', None, 'Weights directory.')
flags.DEFINE_string('img_path', None, 'Image path.')
flags.DEFINE_string('save_path', None, 'Save output path.')
flags.DEFINE_bool('save_raw_output', False, 'Save raw output.')

FLAGS = flags.FLAGS


def main(argv):
  del argv

  config = base_config.get_config(model_name='boundary_attention',
                                  dataset_name='testing',
                                  input_size=(FLAGS.height, FLAGS.width, 3))

  apply_jitted, trained_params = train_utils.make_apply(config,
                                                        FLAGS.weights_dir)

  im_real = np.array(
      PIL.Image.open(
          gfile.GFile(FLAGS.img_path,
                      'rb')).resize((FLAGS.height, FLAGS.width))) / 255.0

  im_use = np.expand_dims(im_real.transpose(2, 0, 1)[:3, :, :], axis=0)

  outputs = apply_jitted(trained_params['params'], im_use)

  viz_utils.visualize_outputs(im_use, outputs)

  plt.savefig(gfile.GFile(FLAGS.save_path + '/output.png', 'wb'), format='png')

  if FLAGS.save_raw_output:
    pickle.dump(outputs,
                gfile.GFile(FLAGS.save_path + '/raw_output.pkl', 'wb'))


if __name__ == '__main__':
  app.run(main)
