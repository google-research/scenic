# Input pipeline from BigTransfer

The code for preprocessing ops in this directory is ported to Scenic from
the implementation of the input pipeline in [Task Adaptation Benchmark](https://arxiv.org/pdf/1910.04867.pdf)
which was also used in the [BigTransfer](https://arxiv.org/abs/1912.11370)
paper. This code is maintained, and was originally implemented, by
[lbeyer@google.com](mailto:lbeyer@google.com) and
[akolesnikov@google.com](mailto:akolesnikov@google.com).

This code allows the user to specify composite preprocessing functions using
strings.  For instance, the following string:
`inception_crop|resize(256)|random_crop(240)|flip_lr|-1_to_1` will be
transformed to a function that makes inception-style crop of the input image,
resizes it to 256x256 size, makes a random crop of size 240x240, flips image
horizontally (with 50% chance) and, finally, transforms it to a range [-1, 1].

If you use this pipeline (i.e., Scenic `bit` datasets), please make sure to cite
the following papers:

```
@article{zhai2019large,
  title={A large-scale study of representation learning with the visual task adaptation benchmark},
  author={Zhai, Xiaohua and Puigcerver, Joan and Kolesnikov, Alexander and Ruyssen, Pierre and Riquelme, Carlos and Lucic, Mario and Djolonga, Josip and Pinto, Andre Susano and Neumann, Maxim and Dosovitskiy, Alexey and others},
  journal={arXiv preprint arXiv:1910.04867},
  year={2019}
}
```
and
```
@inproceedings{kolesnikov2020big,
  title={Big transfer (bit): General visual representation learning},
  author={Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Puigcerver, Joan and Yung, Jessica and Gelly, Sylvain and Houlsby, Neil},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part V 16},
  pages={491--507},
  year={2020},
}
```
