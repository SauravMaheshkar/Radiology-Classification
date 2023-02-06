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
        dim (int): no of dimensions for the Affine and MLP layers
        num_patches (int): number of patches
        layerscale (float): value for the layerscale
    """

    dim: int = 512
    num_patches: int = 16
    layerscale: float = 0.1

    def setup(self) -> None:
        # Affine Layers
        self.affine_1 = Affine(dim=self.dim)
        self.affine_2 = Affine(dim=self.dim)

        # Linear Layer
        self.linear = nn.Dense(features=self.num_patches)

        # LayerScale Parameter
        self.layerscale_val = self.param(
            "layerscale_crosspatch", full, self.dim, self.layerscale
        )

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for CrossPatchSubLayer"""

        # Output from Affine Layer 1
        transform = self.affine_1(x)

        # Transpose the Affine Layer 1
        transposed_transform = jnp.transpose(transform, axes=(0, 2, 1))

        # Feed into Linear Layer
        linear_transform = self.linear(transposed_transform)

        # Tranpose the output from Linear Layer
        transposed_linear = jnp.transpose(linear_transform, axes=(0, 2, 1))

        # Feed into Affine Layer 2
        affine_output = self.affine_2(transposed_linear)

        # Skip-Connection with LayerScale
        return x + affine_output * self.layerscale_val


class CrossChannelSubLayer(nn.Module):
    """
    A Flax linen Module consisting of two Affine element-wise transformations,
    MLP and Skip Connection

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): no of dimensions for the Affine and MLP layers
        layerscale (float): value for the layerscale
        expansion_factor (int): expansion factor of the MLP block
    """

    dim: int = 512
    layerscale: float = 0.1
    expansion_factor: int = 4

    def setup(self) -> None:
        # Affine Layers
        self.affine_1 = Affine(dim=self.dim)
        self.affine_2 = Affine(dim=self.dim)

        # MLP Block
        self.mlp = nn.Sequential(
            [
                nn.Dense(features=self.expansion_factor * self.dim),
                GeLU(),
                nn.Dense(features=self.dim),
            ]
        )

        # LayerScale Parameter
        self.layerscale_val = self.param(
            "layerscale_crosschannel", full, self.dim, self.layerscale
        )

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for CrossChannelSubLayer"""
        # Output from Affine Layer 1
        transform = self.affine_1(x)

        # Feed into the MLP Block
        mlp_output = self.mlp(transform)

        # Output from Affine Layer 2
        affine_output = self.affine_2(mlp_output)

        # Skip-Connection with LayerScale
        return x + affine_output * self.layerscale_val


class ResMLPLayer(nn.Module):
    """
    A Flax linen Module consisting of the CrossPatchSubLayer and CrossChannelSubLayer

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        num_patches (int): no of patches
        dim (int): dimensionality for the Affine and MLP layers
        depth (int): number of blocks of the ResMLP Layer
        expansion_factor (int): expansion factor of the MLP block
    """

    num_patches: int
    dim: int = 512
    depth: int = 12
    expansion_factor: int = 4

    def setup(self) -> None:
        # Determine Value of LayerScale based on the depth
        if self.depth <= 18:
            self.layerscale = 0.1
        elif self.depth > 18 and self.depth <= 24:
            self.layerscale = 1e-5
        else:
            self.layerscale = 1e-6

        self.crosspatch = CrossPatchSubLayer(
            dim=self.dim, num_patches=self.num_patches, layerscale=self.layerscale
        )
        self.crosschannel = CrossChannelSubLayer(
            dim=self.dim,
            layerscale=self.layerscale,
            expansion_factor=self.expansion_factor,
        )

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for ResMLPLayer"""

        # Cross-Patch Sublayer
        crosspatch_ouptput = self.crosspatch(x)

        # Cross-Channel Sublayer
        return self.crosschannel(crosspatch_ouptput)


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
    in_channels: int = 3
    patch_size: int = 16
    num_classes: int = 10
    image_size: int = 224
    expansion_factor: int = 4

    def setup(self) -> None:
        # Attributes
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Patch Projector
        self.patch_projector = nn.Conv(
            features=self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
        )

        # ResMLP Layers
        self.blocks = nn.Sequential(
            [
                ResMLPLayer(
                    dim=self.dim,
                    depth=self.depth,
                    num_patches=self.num_patches,
                    expansion_factor=self.expansion_factor,
                )
                for _ in range(self.depth)
            ]
        )

        # Fully Connected Layer
        self.fully_connected = nn.Dense(features=self.num_classes)

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for ResMLP"""

        # Get the Patch Embeddings
        x = self.patch_projector(x)
        x = x.reshape(-1, self.num_patches, self.dim)

        # Feed into ResMLP Layers
        x = self.blocks(x)

        # Get the mean of the patches
        output = jnp.mean(x, axis=1)

        # Feed into Classification Head
        return self.fully_connected(output)
