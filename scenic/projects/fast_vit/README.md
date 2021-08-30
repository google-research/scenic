## FastViT
FastViT is a project that explores ideas around making ViT model fast and
scalable when applied on high-resolution images.

Given that the tokenizer in the vanilla ViT is simply extracting patches
and embedding them, the number of input tokens to the Transformer encoder (which
is in fact number of patches) depends on the input resolution. So, the
computational cost of ViT is directly tied to the input resolution and given
the quadratic complexity of the self-attention mechanism, applying ViT on
mega-pixel images or any input that requires encoding a long sequence of
patches, e.g., videos, becomes extremely expensive.

To tackle this issue we can use [efficient Transformers](
https://arxiv.org/abs/2009.06732), instead of dot-product attention or we can
reduce the spatial dimensions using pooling mechanisms to reduce the number of
tokens, e.g. having an encoder with a pyramid scheme.

This project is about exploring these ideas and compare different variants in
terms of performance-compute trade-off.

For any question, feel free to reach out to
[Mostaf Dehghani](mailto:dehghani@google.com),
[Yi Tay](mailto:yitay@google.com), or
[Alexey Gritsenko](mailto:agritsenko@google.com).
