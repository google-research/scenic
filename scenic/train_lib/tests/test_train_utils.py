# Copyright 2026 The Scenic Authors.
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

"""Unit tests for training utility functions in train_lib.train_utils."""

import os
from unittest import mock

from absl.testing import absltest
from clu import metric_writers
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
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

  def test_chrono_load_readonly_numpy_scalars(self):
    """Tests Chrono.load and Chrono.tick when restored leaves are read-only."""
    chrono = train_utils.Chrono(example_type="img", warmup=1)
    chrono.inform(first_step=0, total_steps=10, global_bs=8, steps_per_epoch=5)
    ckpt = jax.device_get(jax.tree_util.tree_map(jnp.asarray, chrono.save()))
    chrono.load(ckpt)

    writer = metric_writers.LoggingWriter()
    notes = []
    chrono.tick(step=1, writer=writer, write_note=notes.append)
    chrono.tick(step=2, writer=writer, write_note=notes.append)

    self.assertEqual(chrono.accum_examples_seen, 16)
    self.assertIsInstance(chrono.accum_examples_seen, int)
    self.assertIsInstance(chrono.accum_program_time, float)


class CheckpointTest(absltest.TestCase):
  """Tests checkpoint saving and restoration."""

  def test_legacy_msgpack_checkpoint_restores_when_orbax_enabled(self):
    """Tests restoring a legacy MsgPack checkpoint when Orbax is enabled."""
    workdir = self.create_tempdir().full_path
    prev_orbax_flag = flax.config.flax_use_orbax_checkpointing
    self.addCleanup(
        flax.config.update, "flax_use_orbax_checkpointing", prev_orbax_flag
    )

    flax.config.update("flax_use_orbax_checkpointing", False)
    saved_state = train_utils.TrainState(
        global_step=5,
        params={"w": jnp.array([5.0, 6.0], dtype=jnp.float32)},
        model_state={},
        metadata={"chrono": {"accum_examples_seen": 40}},
    )
    train_utils.save_checkpoint(workdir, saved_state)
    self.assertFalse(
        checkpoints._is_orbax_checkpoint(os.path.join(workdir, "checkpoint_5"))  # pylint: disable=protected-access
    )

    flax.config.update("flax_use_orbax_checkpointing", True)
    target_state = train_utils.TrainState(
        global_step=0,
        params={"w": jnp.zeros((2,), dtype=jnp.float32)},
        model_state={},
        metadata={"chrono": {"accum_examples_seen": 0}},
    )
    restored_state, step = train_utils.restore_checkpoint(workdir, target_state)
    self.assertEqual(step, 5)
    np.testing.assert_allclose(
        restored_state.params["w"], np.array([5.0, 6.0], dtype=np.float32)
    )
    self.assertEqual(
        restored_state.metadata["chrono"]["accum_examples_seen"], 40
    )


if __name__ == "__main__":
  absltest.main()
