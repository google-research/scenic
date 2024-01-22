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

"""Unit tests for functions in model_utils.py."""
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import model_utils


class SimpleGatherTest(parameterized.TestCase):
  """Test simple_gather()."""

  def test_simple_gather_ndarray(self):
    """Test against manually specified target when idx is a nd-array."""
    x = jnp.array(np.random.normal(size=(2, 3, 5)), dtype=jnp.float32)
    idx = jnp.array([[1, 0, 2], [2, 1, 0]], dtype=jnp.int32)
    y = model_utils.simple_gather(x, idx)
    y_target = jnp.stack([
        jnp.stack([x[0, 1], x[0, 0], x[0, 2]]),
        jnp.stack([x[1, 2], x[1, 1], x[1, 0]])])

    self.assertSequenceAlmostEqual(y.flatten(), y_target.flatten())


class LossTest(parameterized.TestCase):
  """Test various loss functions in model_utils."""

  def test_weighted_l1_loss(self):
    """Test weighted_l1_loss against a manually specified target."""
    x = jnp.array([[0.1, 0.3], [-1.0, 0.2]], dtype=jnp.float32)
    y = jnp.array([[0.5, -1.3], [0.9, 1.2]], dtype=jnp.float32)

    out1 = model_utils.weighted_l1_loss(x, y, reduction=None)
    out1_target = jnp.array([[0.4, 1.6], [1.9, 1.0]], dtype=jnp.float32)
    self.assertSequenceAlmostEqual(
        out1.flatten(), out1_target.flatten(), places=5)

    out2 = model_utils.weighted_l1_loss(x, y, reduction='mean').item()
    out2_target = 4.9 / 4
    self.assertAlmostEqual(out2, out2_target, places=5)

  def test_weighted_box_l1_loss(self):
    """Test weighted_box_l1_loss against manually specified targets."""
    x1 = jnp.array([[0.1, 0.3, 0.9, 0.8]], dtype=jnp.float32)
    y1 = jnp.array([[0.5, 0.1, 0.9, 0.7]], dtype=jnp.float32)

    out1 = model_utils.weighted_box_l1_loss(x1, y1)
    out1_target = jnp.array([[0.4, 0.2, 0, 0.1]], dtype=jnp.float32)
    self.assertSequenceAlmostEqual(
        out1.flatten(), out1_target.flatten(), places=5)

    out2 = model_utils.weighted_box_l1_loss(x1, y1, reduction='mean').item()
    out2_target = jnp.mean(out1_target).item()
    self.assertAlmostEqual(out2, out2_target, places=5)

    out3 = model_utils.weighted_box_l1_loss(x1, y1, tight=False)
    out3_target = jnp.array([[0.4, 0.0, 0.0, 0.1]], dtype=jnp.float32)
    self.assertSequenceAlmostEqual(
        out3.flatten(), out3_target.flatten(), places=5)

  def test_weighted_sigmoid_cross_entropy(self):
    """Tests weighted_sigmoid_cross_entropy."""

    logits = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    labels = jnp.array([[0, 1, 1], [1, 0, 1]], dtype=jnp.float32)
    sigmoid = jax.nn.sigmoid
    log = jnp.log

    loss = model_utils.weighted_sigmoid_cross_entropy(logits, labels)
    gt_loss = jnp.array([[
        -log(1 - sigmoid(1.)), -log(sigmoid(2.)), -log(sigmoid(3.))
    ], [-log(sigmoid(4.)), -log(1 - sigmoid(5.)), -log(sigmoid(6.))]
                        ]) / np.prod(labels.shape[:-1])
    self.assertSequenceAlmostEqual(
        loss.flatten(), gt_loss.sum().flatten(), places=3)

    example_weights = jnp.array([1., 0.])
    loss = model_utils.weighted_sigmoid_cross_entropy(
        logits, labels, weights=example_weights)
    gt_loss = jnp.array([[
        -log(1 - sigmoid(1.)), -log(sigmoid(2.)), -log(sigmoid(3.))
    ], [0., 0., 0.]]) / example_weights.sum() + 1e-9
    self.assertSequenceAlmostEqual(
        loss.flatten(), gt_loss.sum().flatten(), places=3)

    label_weights = jnp.array([1., 2., 3.])
    loss = model_utils.weighted_sigmoid_cross_entropy(
        logits, labels, label_weights=label_weights)
    gt_loss = jnp.array([[
        -log(1 - sigmoid(1.)), -2 * log(sigmoid(2.)), -3 * log(sigmoid(3.))
    ], [-log(sigmoid(4.)), -2 * log(1 - sigmoid(5.)), -3 * log(sigmoid(6.))]
                        ]) / np.prod(labels.shape[:-1])
    self.assertSequenceAlmostEqual(
        loss.flatten(), gt_loss.sum().flatten(), places=3)

    loss = model_utils.weighted_sigmoid_cross_entropy(
        logits, labels, weights=example_weights, label_weights=label_weights)
    gt_loss = jnp.array([[
        -log(1 - sigmoid(1.)), -2 * log(sigmoid(2.)), -3 * log(sigmoid(3.))
    ], [0., 0., 0.]]) / example_weights.sum() + 1e-9
    self.assertSequenceAlmostEqual(
        loss.flatten(), gt_loss.sum().flatten(), places=3)

    # Label weights can actually be any shape that is broadcastable to the
    # shape of logits.
    label_weights = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    loss = model_utils.weighted_sigmoid_cross_entropy(
        logits, labels, weights=example_weights, label_weights=label_weights)
    gt_loss = jnp.array([[
        -log(1 - sigmoid(1.)), -2 * log(sigmoid(2.)), -3 * log(sigmoid(3.))
    ], [0., 0., 0.]]) / example_weights.sum() + 1e-9
    self.assertSequenceAlmostEqual(
        loss.flatten(), gt_loss.sum().flatten(), places=3)

    with self.assertRaises(ValueError):
      label_weights = jnp.array([1., 2., 3., 4.])
      loss = model_utils.weighted_sigmoid_cross_entropy(
          logits, labels, label_weights=label_weights)

  def test_focal_sigmoid_cross_entropy(self):
    """Tests focal_sigmoid_cross_entropy."""
    logits = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    labels = jnp.array([[0, 1, 1], [1, 0, 1]], dtype=jnp.float32)
    sigmoid = jax.nn.sigmoid
    log = jnp.log

    a = 0.25
    g = 2.
    loss = model_utils.focal_sigmoid_cross_entropy(
        logits, labels, alpha=a, gamma=g)

    gt_loss = jnp.array(
        [[-log(1 - sigmoid(1.)), -log(sigmoid(2.)), -log(sigmoid(3.))],
         [-log(sigmoid(4.)), -log(1 - sigmoid(5.)), -log(sigmoid(6.))]])
    focal_factor = jnp.array([[
        (1 - a) * sigmoid(1.)**g, a * sigmoid(-2.)**g, a * sigmoid(-3.)**g
    ], [a * sigmoid(-4.)**g, (1 - a) * sigmoid(5.)**g, a * sigmoid(-6.)**g]])
    self.assertSequenceAlmostEqual(
        loss.flatten(), (gt_loss * focal_factor).flatten(), places=3)

  def test_dice_loss(self):
    """Tests the correctness of the segmentation dice loss."""
    # Create test targets:
    batch, num_objects, h, w = 1, 2, 128, 128
    stride = 2
    targets = np.zeros((batch, num_objects, h, w), dtype=np.float32)
    targets[0, 0, :64, :64] = 1.0  # Add object in top left of image.
    targets[0, 1, 64:, 64:] = 1.0  # Add object in bottom right of image.
    input_shape = batch, num_objects, h // stride, w // stride

    # Test perfect predictions:
    inputs = np.zeros(input_shape, dtype=np.float32)
    inputs[0, 0, :64 // stride, :64 // stride] = 1.0
    inputs[0, 1, 64 // stride:, 64 // stride:] = 1.0
    inputs = (inputs - 0.5) * 1e6  # Inputs will be passed through sigmoid.
    loss = model_utils.dice_loss(
        jnp.array(inputs), jnp.array(targets), interpolation='nearest')
    np.testing.assert_array_almost_equal(loss, [[0.0, 0.0]], decimal=3)

    # Test one half-overlapping prediction:
    inputs = np.zeros(input_shape, dtype=np.float32)
    inputs[0, 0, 32 // stride:(32 + 64) // stride, :64 // stride] = 1.0
    inputs[0, 1, 64 // stride:, 64 // stride:] = 1.0
    inputs = (inputs - 0.5) * 1e6  # Inputs will be passed through sigmoid.
    loss = model_utils.dice_loss(
        jnp.array(inputs), jnp.array(targets), interpolation='nearest')
    np.testing.assert_array_almost_equal(loss, [[0.5, 0.0]], decimal=3)

    # Test one non-overlapping prediction:
    inputs = np.zeros(input_shape, dtype=np.float32)
    inputs[0, 0, 64 // stride:, 64 // stride:] = 1.0
    inputs[0, 1, 64 // stride:, 64 // stride:] = 1.0
    inputs = (inputs - 0.5) * 1e6  # Inputs will be passed through sigmoid.
    loss = model_utils.dice_loss(
        jnp.array(inputs), jnp.array(targets), interpolation='nearest')
    np.testing.assert_array_almost_equal(loss, [[1.0, 0.0]], decimal=3)

    # Test all-pairs with different instance numbers:
    inputs = np.zeros((batch, 3, h // stride, w // stride), dtype=np.float32)
    inputs[0, 0, :64 // stride, :64 // stride] = 1.0
    inputs[0, 1, 32 // stride:(32 + 64) // stride, :64 // stride] = 1.0
    inputs[0, 2, 64 // stride:, 64 // stride:] = 1.0
    inputs = (inputs - 0.5) * 1e6  # Inputs will be passed through sigmoid.
    loss = model_utils.dice_loss(
        jnp.array(inputs), jnp.array(targets), interpolation='nearest',
        all_pairs=True)
    self.assertTupleEqual(loss.shape, (1, 3, 2))  # [b, n_pred, n_true]
    np.testing.assert_array_almost_equal(loss, [[[0.0, 1.0],
                                                 [0.5, 1.0],
                                                 [1.0, 0.0]]], decimal=3)

  def test_weighted_square_error(self):
    """Tests implementation of squared error."""

    predictions = jnp.array([
        [
            [1.0, 3.0, 5.0, 6.0],
            [3.0, 5.0, 11.0, 10.0],
            [9.0, 10.0, 11.0, 12.0],
            [14.0, 13.0, 14.0, 17.0],
        ],
        [
            [17.0, 18.0, 21.0, 22.0],
            [20.0, 19.0, 24.0, 25.0],
            [27.0, 29.0, 30.0, 32.0],
            [27.0, 28.0, 33.0, 32.0],
        ],
    ])
    targets = jnp.arange(1, 33).reshape(2, 4, 4)

    # Without specifying axis, this will be over the last two dimensions.
    loss = model_utils.weighted_mean_squared_error(predictions, targets)
    expected_loss = jnp.mean(jnp.array([38.0, 70.0]))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    # Test by specifying axes as a tuple. The following are all equivalent to
    # the previous test.
    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   axis=(1, 2))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   axis=(-1, -2))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   axis=(2, 1))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    # Test by computing loss over a single axis only.
    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   axis=-1)
    expected_loss = jnp.mean(jnp.array([[9, 25, 0, 4],
                                        [8, 12, 38, 12]]))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   axis=2)
    self.assertAlmostEqual(loss, expected_loss, places=5)

    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   axis=1)
    expected_loss = jnp.mean(jnp.array([[5, 3, 21, 9],
                                        [9, 22, 18, 21]]))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    # Test with loss weights.
    weights = jnp.array([[1, 1, 1, 0], [0, 1, 1, 0]])
    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   weights, axis=-1)
    expected_loss = jnp.mean(jnp.array([9, 25, 12, 38, 0]))
    self.assertAlmostEqual(loss, expected_loss, places=5)

    weights = jnp.array([1, 0])
    loss = model_utils.weighted_mean_squared_error(predictions, targets,
                                                   weights, axis=-1)
    expected_loss = jnp.mean(jnp.array([9, 25, 0, 4]))
    self.assertAlmostEqual(loss, expected_loss, places=5)


class MetricTest(parameterized.TestCase):
  """Tests the metric computation related utilities."""

  def is_valid(self, t, value_name):
    """Helper function to assert that tensor `t` does not have `nan`, `inf`."""
    self.assertFalse(
        jnp.isnan(t).any(), msg=f'Found nan\'s in {t} for {value_name}')
    self.assertFalse(
        jnp.isinf(t).any(), msg=f'Found inf\'s in {t} for {value_name}')

  def test_weighted_topk_correctly_classified(self):
    """Tests the topk accuracy computation."""
    batch_size = 512
    num_of_classes = 100
    logits = jnp.array(
        np.random.normal(size=(batch_size, num_of_classes)), dtype=jnp.float32)
    labels = jnp.array(np.random.randint(num_of_classes, size=(batch_size,)))

    one_hot_targets = common_utils.onehot(labels, logits.shape[-1])
    classification_accuracy = model_utils.weighted_correctly_classified(
        logits, one_hot_targets)
    top_one_accuracy = model_utils.weighted_topk_correctly_classified(
        logits, one_hot_targets, k=1)
    self.assertSequenceAlmostEqual(
        classification_accuracy.flatten(), top_one_accuracy.flatten())

    top_n_accuracy = model_utils.weighted_topk_correctly_classified(
        logits, one_hot_targets, k=num_of_classes)
    self.assertEqual(jnp.mean(top_n_accuracy), 1)

    # computes using numpy
    top_5_accuracy = model_utils.weighted_topk_correctly_classified(
        logits, one_hot_targets, k=5)
    top5_pred = np.argsort(
        np.reshape(logits, [-1, num_of_classes]), axis=1)[:, -5:]
    y_true = np.array(labels)
    top5_pred = np.reshape(top5_pred, [-1, 5])
    y_true = np.reshape(y_true, [-1])
    np_top_accuracy = np.array(
        [y_true[i] in top5_pred[i, :] for i in range(len(y_true))])
    self.assertSequenceAlmostEqual(top_5_accuracy.flatten(),
                                   np_top_accuracy.flatten())

  def test_weighted_recall(self):
    """Tests the topk recall computation."""

    logits = np.array([[[2, 3, 4],
                        [4, 3, 2],
                        [4, 2, 3],
                        [3, 2, 4],
                        [4, 2, 3],
                        ]])
    labels = np.array([[[1, 1, 0],
                        [1, 1, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0]
                        ]])

    batch_size = 8
    logits = jnp.tile(logits, [batch_size, 1, 1])
    labels = jnp.tile(labels, [batch_size, 1, 1])

    recall = model_utils.weighted_recall(logits, labels)
    recall_expected = np.array([[1/2, 1., 1., 0., 0.]] * batch_size)
    self.assertSequenceAlmostEqual(
        recall.flatten(), recall_expected.flatten())

  @parameterized.parameters(itertools.product([1., 0.], [1., 0.]))
  def test_weighted_top_one_correctly_classified(self, label_multiplier,
                                                 weight_multiplier):
    """Tests the top1 correct computation."""
    batch_size = 512
    num_of_classes = 100
    logits = jnp.array(np.random.normal(
        size=(batch_size, 50, num_of_classes)), dtype=jnp.float32)
    labels = jnp.array(np.random.randint(
        0, 2, size=(batch_size, 50, num_of_classes)))
    labels *= label_multiplier

    weights = jnp.ones(shape=(batch_size,), dtype=jnp.float32)
    weights *= weight_multiplier

    is_correct_array = model_utils.weighted_top_one_correctly_classified(
        logits, labels, weights=weights)
    num_correct = jnp.sum(is_correct_array)
    is_correct_array_ref = model_utils.weighted_topk_correctly_classified(
        logits, labels, weights, k=1)

    np.testing.assert_array_almost_equal(
        is_correct_array, is_correct_array_ref)
    np.testing.assert_equal(np.sum(is_correct_array),
                            np.sum(is_correct_array_ref))

    self.is_valid(num_correct, 'Number of correctly classified')

  @parameterized.parameters(itertools.product([1., 0.], [1., 0.]))
  def test_weighted_unnormalized_sigmoid_cross_entropy(self, label_multiplier,
                                                       weight_multiplier):
    """Tests the unnormalized sigmoid cross entropy computation."""
    batch_size = 512
    num_of_classes = 100
    logits = jnp.array(
        np.random.normal(size=(batch_size, num_of_classes)), dtype=jnp.float32)
    labels = jnp.array(np.random.randint(0, 2,
                                         size=(batch_size, num_of_classes)))
    labels *= label_multiplier

    weights = jnp.ones(shape=(batch_size,), dtype=jnp.float32)
    weights *= weight_multiplier

    loss_array = model_utils.weighted_unnormalized_sigmoid_cross_entropy(
        logits, labels, weights=weights)
    loss_sum = jnp.sum(loss_array)

    self.is_valid(loss_sum, 'Loss value')

  @parameterized.parameters(itertools.product([1., 0.], [1., 0.]))
  def test_weighted_unnormalized_softmax_cross_entropy(self, label_multiplier,
                                                       weight_multiplier):
    """Tests the unnormalized softmax cross entropy computation."""
    batch_size = 512
    num_of_classes = 100
    logits = jnp.array(
        np.random.normal(size=(batch_size, num_of_classes)), dtype=jnp.float32)
    labels = jnp.array(
        np.random.randint(0, 2, size=(batch_size, num_of_classes)))
    labels *= label_multiplier

    weights = jnp.ones(shape=(batch_size,), dtype=jnp.float32)
    weights *= weight_multiplier

    loss_array = model_utils.weighted_unnormalized_softmax_cross_entropy(
        logits, labels, weights=weights)
    loss_sum = jnp.sum(loss_array)

    self.is_valid(loss_sum, 'Loss value')


if __name__ == '__main__':
  absltest.main()
