"""PyTorch Implementation of MLP-Mixer"""

import torch
from einops.layers.torch import Rearrange
from torch import nn


class MLPBlock(nn.Module):
    """
    A PyTorch implementation of the MLPBlock

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        mlp_dim (int): no of dimensions for the MLP layers
    """

    def __init__(self, mlp_dim: int = 512, **kwargs) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.mlp_dim = mlp_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLPBlock"""

        linear_output = nn.Linear(
            in_features=inputs.shape[-1], out_features=self.mlp_dim
        )(inputs)
        activation_output = nn.functional.gelu(linear_output)
        return nn.Linear(in_features=self.mlp_dim, out_features=inputs.shape[-1])(
            activation_output
        )


class MixerBlock(nn.Module):
    """
    A PyTorch implementation of the MixerBlock consisting
    of a Token Mixing and Channel Mixing sublayer

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        tokens_mlp_dim (int): no of dimensions for the token mixing MLP layers
        channels_mlp_dim (int): no of dimensions for the channel mixing MLP layers
    """

    def __init__(
        self,
        tokens_mlp_dim: int = 384,
        channels_mlp_dim: int = 3072,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

        # Token Mixing
        self.token_mixing = MLPBlock(mlp_dim=self.tokens_mlp_dim)

        # Channel Mixing
        self.channel_mixing = MLPBlock(mlp_dim=self.channels_mlp_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for MixerBlock"""

        # Token Mixing
        prenorm = nn.LayerNorm(inputs.shape[1:])(inputs)
        prenorm_transposed = prenorm.transpose(1, 2)
        mlp_output = self.token_mixing(prenorm_transposed)
        token_mixing_output = inputs + mlp_output.transpose(1, 2)

        # Channel Mixing
        prenorm = nn.LayerNorm(token_mixing_output.shape[1:])(token_mixing_output)
        mlp_output = self.channel_mixing(prenorm)
        channel_mixing_output = token_mixing_output + mlp_output

        return channel_mixing_output


class MLPMixer(nn.Module):
    """
    A PyTorch implementation of the MLP-Mixer

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        in_channels (int): no of channels in the input image
        image_size (int): size of the input image
        patch_size (int): size of the patches
        hidden_dim (int): no of dimensions for the hidden layer
        num_classes (int): no of classes for classification
        num_blocks (int): no of MixerBlocks
        tokens_mlp_dim (int): no of dimensions for the token mixing MLP layers
        channels_mlp_dim (int): no of dimensions for the channel mixing MLP layers
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 768,
        num_classes: int = 10,
        num_blocks: int = 8,
        tokens_mlp_dim: int = 384,
        channels_mlp_dim: int = 3072,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

        # Patch Embedding
        self.patch_projector = nn.Sequential(
            Rearrange("b h w c-> b c h w"),  # channels last to channels first
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )

        # Mixer Blocks
        self.mixer_blocks = nn.Sequential(
            *[
                MixerBlock(
                    tokens_mlp_dim=self.tokens_mlp_dim,
                    channels_mlp_dim=self.channels_mlp_dim,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Classification Head
        self.classification_head = nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_classes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for MLPMixer"""

        # Get the Patch Embeddings
        patches = self.patch_projector(inputs)

        # Feed into Mixer Blocks
        mixerblock_output = patches
        for mixer_block in self.mixer_blocks:
            mixerblock_output = mixer_block(mixerblock_output)

        # Layer Normalization
        layernorm_output = nn.LayerNorm(mixerblock_output.shape[1:])(mixerblock_output)

        # Get the mean of Mixer Block Output
        mean_output = torch.mean(layernorm_output, dim=1)

        # Feed into Classification Head
        classification_head_output = self.classification_head(mean_output)

        return classification_head_output
