# Universal Transformer

>This repo is the reimplementation of Universal Transformers in JAX. We also
include implementation of a "Universal Vision Transformer" wich is a ViT with
dynamic halting mechanism.

The Universal Transformer is an extension to the Transformer models which
combines the parallelizability and global receptive field of the Transformer
model with the recurrent inductive bias of RNNs, which seems to be better
suited to a range of algorithmic and natural language understanding
sequence-to-sequence problems.
Besides, as the name implies, in contrast to the standard Transformer,
under certain assumptions the Universal Transformer can be shown to be
computationally universal.

<img src="fig/UTtransformer.gif" alt="Universal Transformer" width="600"/>

### Universal Transformer: A Concurrent-Recurrent Sequence Model
In the standard Transformer, we have a "fixed" stack of Transformer blocks,
where each block is applied to all the input symbols in parallel.
In the Universal Transformer, however, instead of having a fixed number of
layers, we iteratively apply a Universal Transformer block
(a self-attention mechanism followed by a recurrent transformation) to
refine the representations of all positions in the sequence in parallel,
during an arbitrary number of steps (which is possible due to the recurrence).

In fact,  Universal Transformer is a recurrent function (not in time,but in
depth) that evolves per-symbol hidden states in parallel, based at each step
on the sequence of previous hidden states. In that sense, UT is similar to
architectures such as the Neural GPU and the Neural Turing Machine.
This gives UTs the attractive computational efficiency of the original
feed-forward Transformer model, but with the added recurrent inductive bias
of RNNs.

Note that when running for a fixed number of steps, the Universal Transformer
is equivalent to a multi-layer Transformer with tied parameters across its
layers.

### Universal Transformer with Dynamic Halting
In sequence processing systems, certain symbols (e.g. some words or phonemes)
are usually more ambiguous than others. It is, therefore, reasonable to
allocate more processing resources to these more ambiguous symbols.

As stated before, the standard Transformer applies the same amount of
computations (fixed number of layers) to all symbols in all inputs.
To address this, Universal Transformer with dynamic halting modulates
the number of computational steps needed to process each input symbol
dynamically based on a scalar pondering value that is predicted by the model at
each step. The pondering values are in a sense the model’s estimation of
how much further computation is required for the input symbols at each
processing step.

Universal Transformer with dynamic halting uses an Adaptive
Computation Time (ACT) mechanism, which was originally proposed for RNNS, to
enable conditional computation.

<img src="fig/AdaptiveUT.gif" alt="Universal Transformer with Adaptive Computation" width="600"/>

More precisely, Universal Transformer with dynamic halting adds a dynamic ACT
halting mechanism to each position in the input sequence. Once the per-symbol
recurrent block halts (indicating a sufficient number of revisions for that
symbol), its state is simply copied to the next step until all blocks halt or
we reach a maximum number of steps.  The final output of the encoder is
then the final layer of representations produced in this way.

## Reference
This code is developed by [Fuzhao Xue](https://xuefuzhao.github.io/) and
[Mostafa Dehghani](https://mostafadehghani.com/).
If you use UT, please cite the paper.
```
@article{dehghani2018universal,
  title={Universal transformers},
  author={Dehghani, Mostafa and Gouws, Stephan and Vinyals, Oriol and Uszkoreit, Jakob and Kaiser, {\L}ukasz},
  journal={arXiv preprint arXiv:1807.03819},
  year={2018}
}
```
