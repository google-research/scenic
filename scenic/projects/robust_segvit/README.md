# Robust segvit

*Robust_segvit* is a pipeline to evaluate the robustness of semantic segmentation models.

Robust_segvit is developed in [JAX](https://github.com/google/jax) and uses [Flax](https://github.com/google/flax).


## Code structure
This code includes several datasets such as: <br>

cityscapes_variants: <br>
  - fishyscapes. <br>
  - cityscapes_c (a corrupted version of cityscapes). <br>

segmentation_datasets: <br>
  - [ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/). <br>
  - ade20k_ind, the ade20k dataset with 3 classes dropped. <br>
  - ade20k_ood, the ade20k with only 3 classes for OOD detection. <br>

segmentation_variants: <br>
  - ade20k_corrupted, the corrupted version of the ade20k dataset. <br>
  - ade20k_ind_c, the corrupted version of the ade20k_ind dataset. <br>

See [uncertainty_baselines/experimental/robust_segvit](https://github.com/google/uncertainty-baselines/experimental/robust_segvit) for examples on how to use these datasets.
