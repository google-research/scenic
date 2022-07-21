# Point Cloud Processing Using Transformers

This repository contains the implementation of
[Point Cloud Transformers](https://arxiv.org/abs/2012.09688) which is a
Transformer-based framework for processing point clouds.

## Architecture

The PCT Encoder uses the following pipeline leveraging 4 self-attention layers.

```
X -> Dense(X) -> SA_1(X) -> SA_2(X) -> SA_3(X) -> SA_4(X)
```

where X is the input, SA_i's are different self-attention layers. The outputs
from the self-attention layers are concatenated and undergoes aggregation (e.g.
max- or mean-pooling) before being used for task-specific applications like
classification and Segmentation.

## Contribution

We have replicated the classification baseline on ModelNet40 using NaivePCT
architecture. This repository contains the implementation of the Naive and Offset
PCT models.

