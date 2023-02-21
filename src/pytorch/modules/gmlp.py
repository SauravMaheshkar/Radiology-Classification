"""PyTorch Implementation of gMLP"""

import torch
from torch import nn


class SpatialGatingUnit(nn.Module):
    """
    A PyTorch implementation of the SpatialGatingUnit
    References:
        - https://arxiv.org/abs/2105.08050v2
    Attributes:
        dim (int): no of dimensions for the SpatialGatingUnit
    """

    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for SpatialGatingUnit"""
        # pylint: disable=invalid-name

        u, v = inputs.chunk(chunks=2, dim=-1)

        v = nn.LayerNorm(normalized_shape=v.shape, eps=1e-6)(v)
        v = v.permute(0, 2, 1)
        v = nn.Linear(in_features=self.dim, out_features=self.dim, bias=True)(v)
        v = v.permute(0, 2, 1)

        return u * v
        # pylint: enable=invalid-name


class gMLPBlock(nn.Module):  # pylint: disable=invalid-name
    """
    A PyTorch implementation of the gMLPBlock
    References:
        - https://arxiv.org/abs/2105.08050v2
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for gMLPBlock"""
        shortcut = inputs
        norm = nn.LayerNorm(normalized_shape=inputs.shape, eps=1e-6)(inputs)
        in_proj = nn.Linear(
            in_features=inputs.shape[-1], out_features=inputs.shape[-1] * 2, bias=True
        )(norm)
        activations = nn.GELU()(in_proj)
        sgu = SpatialGatingUnit(dim=inputs.shape[-2])(activations)
        out_proj = nn.Linear(
            in_features=inputs.shape[-1], out_features=inputs.shape[-1], bias=True
        )(sgu)
        return out_proj + shortcut
