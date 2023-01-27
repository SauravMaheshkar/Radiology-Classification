"""Affine Transformation Layer in PyTorch"""

import torch
from torch import nn

__all__ = ["Affine"]


class Affine(nn.Module):
    """
    A PyTorch Module to perform a Affine Transformation

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): Needed to generate matrices of the appropriate shape
    """

    def __init__(self, dim: int = 512, **kwargs) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim = dim

        # Parameters
        self.alpha = nn.Parameter(torch.ones(1, 1, self.dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, self.dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass through the Affine Transformation Layer"""
        return inputs * self.alpha + self.beta
