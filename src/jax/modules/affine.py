"""Affine Transformation Module in Flax"""

from flax import linen as nn

from jax._src.typing import ArrayLike

# local imports
from src.jax.utils import ones, zeros

__all__ = ["Affine"]


class Affine(nn.Module):
    """
    A Flax linen Module to perform a Affine Transformation

    Attributes:
        dim (int): Needed to generate matrices of the appropriate shape
    """

    dim: int = 512

    def setup(self) -> None:
        self.alpha = self.param("alpha", ones, (1, 1, self.dim))
        self.beta = self.param("beta", zeros, (1, 1, self.dim))

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        return x * self.alpha + self.beta
