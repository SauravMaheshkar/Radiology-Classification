"""Flax Implementation of ResMLP"""

import flax.linen as nn

import jax.numpy as jnp
from jax._src.typing import ArrayLike

# local imports
from src.jax.activations.gelu import GeLU
from src.jax.modules.affine import Affine
from src.jax.utils import full

__all__ = ["CrossPatchSubLayer", "CrossChannelSubLayer", "ResMLPLayer", "ResMLP"]


class CrossPatchSubLayer(nn.Module):
    """
    A Flax linen Module consisting of two Affine element-wise transformations,
    Linear Layer and Skip Connection

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim: dimensionality for the Affine Layer
        patch_size: dimensionality for the Linear Layer
        layerscale: float value for scaling the output
    """

    dim: int = 512
    patch_size: int = 16
    layerscale: float = 0.1

    def setup(self) -> None:
        self.affine_1 = Affine(dim=self.dim)
        self.linear = nn.Dense(features=self.patch_size)
        self.affine_2 = Affine(dim=self.dim)
        self.layerscale_val = self.param(
            "layerscale_crosspatch", full, self.dim, self.layerscale
        )

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:

        # Output from Affine Layer 1
        transform = self.affine_1(x)

        # Transpose the Affine Layer 1
        transposed_transform = jnp.transpose(transform)

        # Feed into Linear Layer
        linear_transform = self.linear(transposed_transform)

        # Tranpose the output from Linear Layer
        transposed_linear = jnp.transpose(linear_transform)

        # Feed into Affine Layer 2
        affine_output = self.affine_2(transposed_linear)

        # Skip-Connection with LayerScale
        output = x + affine_output * self.layerscale_val

        return output


class CrossChannelSubLayer(nn.Module):
    """
    A Flax linen Module consisting of two Affine element-wise transformations,
    MLP and Skip Connection

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim: dimensionality for the Affine Layer and MLP fully-connected layers
        layerscale: float value for scaling the output
        expansion_factor: expansion factor for the MLP
    """

    dim: int = 512
    layerscale: float = 0.1
    expansion_factor: int = 4

    def setup(self) -> None:
        self.affine_1 = Affine(dim=self.dim)
        self.mlp = nn.Sequential(
            [
                nn.Dense(features=self.expansion_factor * self.dim),
                GeLU(),
                nn.Dense(features=self.dim),
            ]
        )
        self.affine_2 = Affine(dim=self.dim)
        self.layerscale_val = self.param(
            "layerscale_crosschannel", full, self.dim, self.layerscale
        )

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:

        # Output from Affine Layer 1
        transform = self.affine_1(x)

        # Feed into the MLP Block
        mlp_output = self.mlp(transform)

        # Output from Affine Layer 2
        affine_output = self.affine_2(mlp_output)

        # Skip-Connection with LayerScale
        output = x + affine_output * self.layerscale_val

        return output


class ResMLPLayer(nn.Module):
    """
    A Flax linen Module consisting of the CrossPatchSubLayer and CrossChannelSubLayer

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        depth: No of ResMLP Layers, needed for determining the layerscale value
        patch_size: dimensionality for the Linear Layer
    """

    dim: int = 512
    depth: int = 12
    patch_size: int = 16

    def setup(self) -> None:

        # Determine Value of LayerScale based on the depth
        if self.depth <= 18:
            self.layerscale = 0.1
        elif self.depth > 18 and self.depth <= 24:
            self.layerscale = 1e-5
        else:
            self.layerscale = 1e-6

        self.crosspatch = CrossPatchSubLayer(
            dim=self.dim, patch_size=self.patch_size, layerscale=self.layerscale
        )
        self.crosschannel = CrossChannelSubLayer(
            dim=self.dim, layerscale=self.layerscale
        )

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:

        crosspatch_ouptput = self.crosspatch(x)
        crosschannel_output = self.crosschannel(crosspatch_ouptput)

        return crosschannel_output


class ResMLP(nn.Module):
    """
    A Flax linen Module for creating the ResMLP architecture

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        depth: Number of ResMLP layers
        patch_size: dimensionality of the patches
        num_classes: No of classes
    """

    dim: int = 512
    depth: int = 12
    patch_size: int = 16
    num_classes: int = 10

    def setup(self) -> None:
        self.patch_projector = nn.Conv(
            features=self.dim,
            kernel_size=[self.patch_size],
            strides=self.patch_size,
        )

        self.blocks = nn.Sequential(
            [
                ResMLPLayer(dim=self.dim, patch_size=self.patch_size, depth=self.depth)
                for _ in range(self.depth)
            ]
        )

        self.fully_connected = nn.Dense(features=self.num_classes)

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        x = self.patch_projector(x)
        x = self.blocks(x)
        output = jnp.mean(x, axis=1)
        return self.fully_connected(output)
