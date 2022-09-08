# Robust segvit

*Robust_segvit* is a codebase to evaluate the robustness of semantic segmentation models.

Robust_segvit is developed in [JAX](https://github.com/google/jax) and uses [Flax](https://github.com/google/flax), [uncertainty_baselines](https://github.com/google/uncertainty-baselines) and [Scenic](https://github.com/google-research/scenic).

## Code structure
See [uncertainty_baselines](https://github.com/google/uncertainty-baselines)/experimental/robust_segvit.

The datasets included are:
[x] cityscapes_variants: fishyscapes and cityscapes_c.
[x] segmentation_datasets: ade20k, ade20k_ind, ade20k_ood.
[x] segmentation_variants: ade20k_c.
