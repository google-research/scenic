# Robust segvit

*Robust_segvit* is a pipeline to evaluate the robustness of semantic segmentation models.

Robust_segvit is developed in [JAX](https://github.com/google/jax) and uses [Flax](https://github.com/google/flax).


## Code structure
This code includes the following datasets: <br>

segmentation_datasets: <br>
  - [cityscapes](https://www.cityscapes-dataset.com/). <br>
  - [ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/). <br>
  - ade20k_ind, same as ade20k dataset with 3 classes dropped. <br>
  - ade20k_ood, fraction of ade20k with 3 dropped classes for evaluation. <br>
  - [street_hazards](https://arxiv.org/abs/1911.11132). <br>

segmentation_variants: <br>
  - cityscapes_corrupted, the corrupted version of the cityscapes dataset. <br>
  - ade20k_corrupted, the corrupted version of the ade20k dataset. <br>
  - ade20k_ind_c, the corrupted version of the ade20k_ind dataset. <br>
  - street_hazards_corrupted, the corrupted version of the street_hazards dataset. <br>

See [uncertainty_baselines/experimental/robust_segvit](https://github.com/google/uncertainty-baselines/experimental/robust_segvit) for examples on how to use these datasets.
