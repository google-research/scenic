"""Normalization functions."""

import flax.linen as nn


NORMALIZATIONS_BY_NAME = {
    'batch': nn.BatchNorm,
    'layer': nn.LayerNorm,
}
