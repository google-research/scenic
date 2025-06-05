# Copyright 2025 The Scenic Authors.
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

"""Unit tests for training utility functions in train_lib.train_utils.

This file covers tests for the Chrono context manager.
"""

from unittest import mock

from absl.testing import absltest
from scenic.train_lib import train_utils


class ChronoPausedTest(absltest.TestCase):
  """Tests the Chrono.paused context manager for correct behavior."""

  @mock.patch("jax.block_until_ready", autospec=True)
  @mock.patch("time.monotonic")
  def test_paused_context_manager_waits_executes_the_code_block_and_resumes(
      self, mock_monotonic, mock_block_until_ready
  ):
    """Tests the Chrono.paused context manager in a normal flow."""
    chrono = train_utils.Chrono()
    before_pause, after_pause, after_resume = 100.0, 101.1, 105.5
    mock_monotonic.side_effect = [before_pause, after_pause, after_resume]
    wait_for_ops = [mock.MagicMock()]  # Dummy operations to await.

    with chrono.paused(wait_for=wait_for_ops):
      mock_block_until_ready.assert_called_once_with(wait_for_ops)
      self.assertEqual(chrono.pause_start, before_pause)

    self.assertIsNone(chrono.pause_start)  # Should be reset by resume
    self.assertEqual(chrono.paused_time, after_pause - before_pause)
    self.assertEqual(mock_monotonic.call_count, 3)  # init, pause, and resume

  @mock.patch("jax.block_until_ready", autospec=True)
  @mock.patch("time.monotonic")
  def test_paused_context_manager_with_exception_calls_resume(
      self, mock_monotonic, mock_block_until_ready
  ):
    """Tests that Chrono.resume is called even if an exception occurs."""
    chrono = train_utils.Chrono()
    before_pause, after_pause, after_resume = 100.0, 101.1, 105.5
    mock_monotonic.side_effect = [before_pause, after_pause, after_resume]
    wait_for_ops = ("dummy_op",)
    custom_exception = ValueError("Test exception inside context")

    # Disable linting since the assertion against the exception must be done
    # within the context manager. The assertions below the context blocks are
    # not affected by the exception, despite the highlighting (or dimming).
    with self.assertRaises(ValueError) as context:  # pylint: disable=g-error-prone-assert-raises
      with chrono.paused(wait_for=wait_for_ops):
        mock_block_until_ready.assert_called_once_with(wait_for_ops)
        self.assertEqual(chrono.pause_start, before_pause)
        raise custom_exception
      self.assertEqual(context.exception, custom_exception)

    self.assertIsNone(chrono.pause_start)  # Should be reset by resume
    self.assertEqual(chrono.paused_time, after_pause - before_pause)
    self.assertEqual(mock_monotonic.call_count, 3)  # init, pause, and resume


if __name__ == "__main__":
  absltest.main()
