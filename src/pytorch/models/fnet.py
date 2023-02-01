"""PyTorch Implementation of FNet"""

import torch
from einops.layers.torch import Rearrange
from torch import nn

__all__ = ["FourierTransformLayer", "FNetFeedForwardLayer", "FNetEncoderBlock", "FNet"]


class FourierTransformLayer(nn.Module):
    """
    A PyTorch implementation of the Fourier Transform Layer

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        patch_size (int): patch size
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Fourier Transform Layer
        self._fourier_transform = torch.fft.fftn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for FourierTransformLayer"""

        return torch.real(self._fourier_transform(inputs))


class FNetFeedForwardLayer(nn.Module):
    """
    A PyTorch implementation of a Feed Forward Layer
    for the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        dim_ff: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for FNetFeedForwardLayer"""

        intermediate_dense_output = nn.Linear(
            in_features=inputs.shape[-1], out_features=self.dim_ff
        )(inputs)
        intermediate_activations = nn.GELU()(intermediate_dense_output)
        dense_output = nn.Linear(
            in_features=self.dim_ff, out_features=inputs.shape[-1]
        )(intermediate_activations)
        return nn.Dropout(p=self.dropout_rate)(dense_output)


class FNetEncoderBlock(nn.Module):
    """
    A PyTorch implementation of the FNet Encoder Block

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        dim_ff: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate

        # Layers
        self.fourier_transform_layer = FourierTransformLayer()
        self.fnet_feed_forward_layer = FNetFeedForwardLayer(
            dim_ff=self.dim_ff, dropout_rate=self.dropout_rate
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for FNetEncoderBlock"""

        mixing_output = self.fourier_transform_layer(inputs)
        mixing_layer_norm = nn.LayerNorm(normalized_shape=inputs.shape[-1], eps=1e-12)(
            inputs + mixing_output
        )
        feed_forward_output = self.fnet_feed_forward_layer(mixing_layer_norm)
        return nn.LayerNorm(normalized_shape=inputs.shape[-1], eps=1e-12)(
            mixing_layer_norm + feed_forward_output
        )


class FNet(nn.Module):
    """
    A PyTorch implementation of the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        num_layers: int = 12,
        num_classes: int = 10,
        dim_ff: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate

        # Patch Embedding
        self.patch_embedding = nn.Sequential(
            Rearrange("b h w c-> b c h w"),  # channels last to channels first
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.dim_ff,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )

        # Encoder Blocks
        self.encoder_blocks = nn.Sequential(
            *[
                FNetEncoderBlock(dim_ff=self.dim_ff, dropout_rate=self.dropout_rate)
                for _ in range(self.num_layers)
            ]
        )

        # Classification Head
        self.classification_head = nn.Linear(
            in_features=self.dim_ff, out_features=self.num_classes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for FNet"""

        # Get the Patch Embeddings
        patches = self.patch_embedding(inputs)

        # Pass through the Encoder Blocks
        encoder_blocks_output = patches
        for encoder_block in self.encoder_blocks:
            encoder_blocks_output = encoder_block(encoder_blocks_output)

        # Get thee mean of Encoder Blocks Output
        mean_encoder_blocks_output = torch.mean(encoder_blocks_output, dim=1)

        # Pass through the Classification Head
        return self.classification_head(mean_encoder_blocks_output)
