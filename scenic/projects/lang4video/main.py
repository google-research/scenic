"""Main file for the Scenic project Lang4Video."""

from clu import metric_writers
from clu import platform
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.lang4video import main_lib


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter) -> None:
  with config.unlocked():
    if 'evaluation_configs' in config:
      # Dicts inside lists/tuples can't be frozen. Plus, this is unused here:
      del config['evaluation_configs']

  # We do this after removing the evaluation configs because they are too many.
  writer.write_hparams(config)

  main_lib.main(rng=rng, config=config, workdir=workdir, writer=writer)

  # Log final checkpoint path as XManager artefact to tell parallel jobs that
  # training is done:
  if jax.process_index() == 0 and (latest_checkpoint :=
                                   checkpoints.latest_checkpoint(workdir)):
    # XManager overwrites artifacts with identical content even if their
    # description is different. As a workaround, we prepend "TRAIN", so that
    # the path can be distinguished from the "Last evaluated checkpoint"
    # artifact written by the evaluator.
    artifact = 'TRAIN' + latest_checkpoint
    platform.work_unit().create_artifact(
        platform.ArtifactType.FILE, artifact,
        main_lib.FINAL_CKPT_ARTIFACT_DESCRIPTION)


if __name__ == '__main__':
  app.run(main=main)
