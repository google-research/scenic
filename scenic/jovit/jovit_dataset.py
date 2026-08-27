# Copyright 2022 The Scenic Authors.
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

"""Tag with JoViT a wrapped dataset.

The dataset accepts three configuration fields:

 * xid: The experiment id of the jovit model.
 * wid: The experiment id of the jovit model.
 * dataset_name: The model of the wrapped dataset.

All other fields are passed directly to the wrapped dataset. The only
requirement is that the 'inputs' key of the wrapped dataset is of shape
[..., h, w, c], s.t., (h, w, c) is compatible with the jovit run.
"""
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.google.xm import xm_utils
from scenic.projects.jovit import model_loader


def tag(jovit_fn, ds_iter):
  for example in ds_iter:
    x = example['inputs']
    y = jovit_fn(x)['pre_logits']
    y = y.reshape((y.shape[:-1] + (1, 1) + y.shape[-1:]))
    example['inputs'] = y
    yield example


@datasets.add_dataset('jovit_preprocessed')
def get_dataset(dataset_configs, *args, **kwargs):
  """Loads the dataset."""
  xid = dataset_configs.get('xid', 47675785)
  wid = dataset_configs.get('wid', 1)
  jovit_config, _ = xm_utils.get_info_from_xmanager(xid, wid)
  jovit_fn = model_loader.jovit(xid=xid, wid=wid)
  wrapped_name = dataset_configs.dataset_name
  builder = datasets.get_dataset(dataset_name=wrapped_name)
  wrapped_dataset = builder(dataset_configs=dataset_configs,
                            *args, **kwargs)
  meta_data = wrapped_dataset.meta_data
  meta_data['input_shape'] = list(meta_data['input_shape'])
  meta_data['input_shape'][-3:] = [1, 1, jovit_config.model.hidden_size]
  return dataset_utils.Dataset(
      train_iter=tag(jovit_fn, wrapped_dataset.train_iter),
      valid_iter=tag(jovit_fn, wrapped_dataset.valid_iter),
      test_iter=tag(jovit_fn, wrapped_dataset.test_iter),
      meta_data=meta_data)
