# Verbs in Action: Improving verb understanding in video-language models

JAX implementation for Verb-Focused Contrastive (VFC) learning of video-text models.
For details, see [`arXiv`](https://arxiv.org/abs/2304.06708).

<img src="vfc.png" width="750" height="180" />


## Training
Like other projects in Scenic, all model parameters, training sets and datasets are specified using [configuration files](configs).

An example command-line to train our Verb-Focused Contrastive (VFC) pre-training on the Spoken Moments in Time dataset using this [config file](configs/vfc.py) is:

```shell
$ python -m scenic.projects.verbs_in_action.main \
  --config=scenic/projects/verbs_in_action/configs/vfc.py \
  --workdir=verb_focused_contrastive/
```

Likewise, you can train a baseline model on the Spoken Moments in Time dataset. Oue baseline is a standard contrastive video-text model and corresponds to the run coined as `Baseline` for example in tables 2, 3 or 6 of our paper.
We follow this [config file](configs/baseline.py) and run:

```shell
$ python -m scenic.projects.verbs_in_action.main \
  --config=scenic/projects/verbs_in_action/configs/baseline.py \
  --workdir=baseline_contrastive/
```

## Citation

If you use the `verbs in action` project, please cite the following BibTeX entry:

```
@article{momeni2023verbs,
  title={Verbs in Action: Improving verb understanding in video-language models},
  author={Momeni, Liliane and Caron, Mathilde and Nagrani, Arsha and Zisserman, Andrew and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2304.06708},
  year={2023}
}
```
