"""Flax Implementation of FNet"""

import flax.linen as nn

import jax
import jax.numpy as jnp
from jax._src.typing import ArrayLike

__all__ = ["FourierTransformLayer", "FNetFeedForwardLayer", "FNetEncoderBlock", "FNet"]


class FourierTransformLayer(nn.Module):
    """
    A Flax linen Module consisting of a Fourier Transform Layer

    References:
        - https://arxiv.org/abs/2105.03824
    """

    @nn.compact
    def __call__(self, inputs: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Forward pass for FourierTransformLayer"""
        return jax.vmap(jnp.fft.fftn)(inputs).real


class FNetFeedForwardLayer(nn.Module):
    """
    A Flax linen Module implementation of a Feed Forward Layer
    for the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        dropout_rate (float): dropout rate

    """

    dim_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs: ArrayLike, training: bool, *args, **kwargs) -> ArrayLike:
        """Forward pass for FNetFeedForwardLayer"""
        intermediate_dense_output = nn.Dense(
            features=self.dim_ff,
            kernel_init=nn.initializers.normal(2e-2),
            bias_init=nn.initializers.normal(2e-2),
            name="intermediate_dense_layer",
        )(inputs)
        dense_activation = nn.gelu(intermediate_dense_output)
        dense_output = nn.Dense(
            features=inputs.shape[-1],
            kernel_init=nn.initializers.normal(2e-2),
            name="output_dense_layer",
        )(dense_activation)
        output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(
            dense_output
        )
        return output


class FNetEncoderBlock(nn.Module):
    """
    A Flax linen Module implementation of an Encoder Block
    for the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824
    """

    fourier_layer: FourierTransformLayer
    feed_forward_layer: FNetFeedForwardLayer

    @nn.compact
    def __call__(self, inputs: ArrayLike, training: bool, *args, **kwargs) -> ArrayLike:
        """Forward pass for FNetEncoderBlock"""

        mixing_output = self.fourier_layer(inputs)
        mixing_layer_norm = nn.LayerNorm(
            epsilon=1e-12, name="mixing_layer_normalization"
        )(inputs + mixing_output)
        feed_forward_output = self.feed_forward_layer(mixing_layer_norm, not training)
        return nn.LayerNorm(epsilon=1e-12, name="output_layer_normalization")(
            mixing_layer_norm + feed_forward_output
        )


class FNet(nn.Module):
    """
    A Flax linen Module implementation of the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        image_size (int): image size
        dim_ff (int): no of dimensions for the Feed Forward Layer
        patch_size (int): patch size
        num_layers (int): no of layers
        num_classes (int): no of classes
        dropout_rate (float): dropout rate
    """

    dim_ff: int
    patch_size: int
    num_layers: int
    num_classes: int
    image_size: int = 224
    dropout_rate: float = 0.1

    def setup(self) -> None:
        # Attributes
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        # Patch Projector
        self.patch_projector = nn.Conv(
            features=self.dim_ff,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            kernel_init=nn.initializers.normal(2e-2),
            bias_init=nn.initializers.normal(2e-2),
            name="patch_projector",
        )

        # Encoder Blocks
        encoder_blocks = []
        for layer in range(self.num_layers):
            encoder_blocks.append(
                FNetEncoderBlock(
                    fourier_layer=FourierTransformLayer(),
                    feed_forward_layer=FNetFeedForwardLayer(
                        dim_ff=self.dim_ff, dropout_rate=self.dropout_rate
                    ),
                    name=f"encoder_block_{layer}",
                )
            )
        self.encoder_blocks = encoder_blocks

        # Fully Connected Layer
        self.classifier = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.normal(2e-2),
            bias_init=nn.initializers.normal(2e-2),
        )

    def __call__(self, inputs: ArrayLike, training: bool, *args, **kwargs) -> ArrayLike:
        """Forward pass for FNet"""

        # Patch Projector
        patches = self.patch_projector(inputs)

        # Reshape
        patches_reshaped = patches.reshape(patches.shape[0], -1, patches.shape[-1])

        # Encoder Blocks
        encoder_output = patches_reshaped
        for encoder_block in self.encoder_blocks:
            encoder_output = encoder_block(encoder_output, training)

        # Pooling
        pooled_output = jnp.mean(encoder_output, axis=1)

        # Feed into Classifier
        output = self.classifier(pooled_output)

        return output
