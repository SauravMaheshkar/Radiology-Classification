"""Flax implementation of GMLP"""

from flax import linen as nn

from jax._src.typing import ArrayLike
from src.jax.utils import ones

__all__ = ["SpatialGatingUnit"]


class SpatialGatingUnit(nn.Module):
    """
    A Flax linen Module to perform a Spatial Gating Unit

    References:
        - https://arxiv.org/abs/2105.08050v2

    Attributes:
        dim (int): no of dimensions for the SpatialGatingUnit
    """

    dim: int = 512

    def setup(self) -> None:
        """Setup the SpatialGatingUnit layer based on the input shape"""
        self.dense = nn.Dense(features=self.dim)

    @nn.compact
    def __call__(self, inputs: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Compute a forward pass through the SpatialGatingUnit"""
        # pylint: disable=invalid-name
        u, v = inputs.split(2, axis=-1)
        v = nn.LayerNorm()(v)
        v = self.dense(v.transpose((0, 2, 1))).transpose((0, 2, 1))
        return u * v
        # pylint: enable=invalid-name


class gMLPBlock(nn.Module):  # pylint: disable=invalid-name
    """
    A Flax linen Module to perform a gMLPBlock

    References:
        - https://arxiv.org/abs/2105.08050v2
    """

    @nn.compact
    def __call__(self, inputs: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Compute a forward pass through the gMLPBlock"""
        shortcut = inputs

        norm = nn.LayerNorm(epsilon=1e-6)(inputs)
        in_proj = nn.Dense(features=inputs.shape[-1] * 2, bias_init=ones)(norm)
        activations = nn.gelu(in_proj)
        sgu = SpatialGatingUnit(dim=inputs.shape[-2])(activations)
        out_proj = nn.Dense(features=inputs.shape[-1])(sgu)

        return out_proj + shortcut
