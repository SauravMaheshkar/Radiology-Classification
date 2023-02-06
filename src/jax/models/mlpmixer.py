"""Flax Implementation of MLP-Mixer"""

import typing
from typing import Optional

import einops
import flax.linen as nn

import jax.numpy as jnp
from jax._src.typing import ArrayLike

__all__ = ["MLPBlock", "MixerBlock", "MLPMixer"]


class MLPBlock(nn.Module):
    """
    A Flax linen Module implementation of the MLPBlock

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        mlp_dim (int): no of dimensions for the MLP layers
    """

    mlp_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for MLPBlock"""

        linear_output = nn.Dense(self.mlp_dim)(x)
        activation_output = nn.gelu(linear_output)
        return nn.Dense(x.shape[-1])(activation_output)


class MixerBlock(nn.Module):
    """
    A Flax linen Module consisting of two MLP Blocks
    for token mixing and channel mixing

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        tokens_mlp_dim (int): no of dimensions for the token mixing MLP layers
        channels_mlp_dim (int): no of dimensions for the channel mixing MLP layers
    """

    tokens_mlp_dim: int = 384
    channels_mlp_dim: int = 3072

    @nn.compact
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for MixerBlock"""

        # Token Mixing
        prenorm = nn.LayerNorm()(x)
        prenorm_transposed = jnp.swapaxes(prenorm, 1, 2)
        mlp_output = MLPBlock(mlp_dim=self.tokens_mlp_dim, name="token_mixing")(
            prenorm_transposed
        )

        # Channel Mixing
        mlp_output_transposed = jnp.swapaxes(mlp_output, 1, 2)
        skip_connection = x + mlp_output_transposed
        postnorm = nn.LayerNorm()(skip_connection)
        return x + MLPBlock(mlp_dim=self.channels_mlp_dim, name="channel_mixing")(
            postnorm
        )


class MLPMixer(nn.Module):
    """
    A Flax linen Module of the MLP-Mixer architecture

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        patch_size (Optional[int]): patch size, defaults to 16
        num_classes (Optional[int]): number of classes, defaults to 10
        num_blocks (Optional[int]): number of blocks, defaults to 12
        hidden_dim (Optional[int]): hidden dimension, defaults to 768
        tokens_mlp_dim (Optional[int]): dimensions for the token mixing layers,
            defaults to 384
        channels_mlp_dim (Optional[int]): dimensions for the channel mixing layers,
            defaults to 3072
        model_name (Optional[str]): name of the model, defaults to "Mixer-B_16"
    """

    patch_size: Optional[int] = 16
    num_classes: Optional[int] = 10
    num_blocks: Optional[int] = 12
    hidden_dim: Optional[int] = 768
    tokens_mlp_dim: Optional[int] = 384
    channels_mlp_dim: Optional[int] = 3072
    model_name: Optional[str] = "Mixer-B_16"

    @nn.compact
    def __call__(self, inputs: ArrayLike, *args, **kwargs) -> ArrayLike:
        # Get the Patch Embeddings
        patches = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            name="stem",
        )(inputs)
        rearranged_patches = einops.rearrange(patches, "n h w c -> n (h w) c")

        # Feed into Mixer Blocks
        mixerblock_output = rearranged_patches
        for _ in range(typing.cast(int, self.num_blocks)):
            mixerblock_output = MixerBlock(
                tokens_mlp_dim=self.tokens_mlp_dim,
                channels_mlp_dim=self.channels_mlp_dim,
            )(mixerblock_output)

        # Layer Normalization
        layernorm_output = nn.LayerNorm(name="pre_head_layer_norm")(mixerblock_output)

        # Get the mean of the patches
        x = jnp.mean(layernorm_output, axis=1)  # pylint: disable=invalid-name

        # Feed into Classification Head
        if self.num_classes:
            x = nn.Dense(  # pylint: disable=invalid-name
                self.num_classes, kernel_init=nn.initializers.zeros, name="head"
            )(x)
        return x
