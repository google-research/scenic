Datasets
==

The training pipeline uses the [DeepMind Video Reader (DMVR)](https://github.com/deepmind/dmvr)
library for pre-processing and data-augmentation.
Futhermore, we assume that datasets are stored in in [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files.

To pre-process a dataset into the required format, please follow the
instructions from DMVR [here](https://github.com/deepmind/dmvr/tree/master/examples).

Once a dataset has been pre-processed, it can easily be used for training by
adding the following snippet to the configuration file:

```python
dataset_configs = ml_collections.ConfigDict()

dataset_configs.base_dir = '/path/to/dataset_root/'
dataset_configs.tables = {
    'train': 'relative_path_to_train_set',
    'validation': 'relative_path_to_validation_set',
    'test': 'relative_path_to_test_set'
}
dataset_configs.examples_per_subset = {
    'train': NUM_TRAIN_EXAMPLES,
    'validation': NUM_VAL_EXAMPLES,
    'test': NUM_TEST_EXAMPLES
}
dataset_configs.num_classes = NUM_CLASSES
```
