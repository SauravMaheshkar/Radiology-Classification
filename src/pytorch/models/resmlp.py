"""PyTorch implementation of ResMLP"""

import torch
from einops.layers.torch import Rearrange
from torch import nn

# local imports
from src.pytorch.modules.affine import Affine


class CrossChannelSubLayer(nn.Module):
    """
    A PyTorch implementation of the CrossChannelSubLayer

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): no of dimensions for the Affine and MLP layers
        layerscale (float): value for the layerscale
        expansion_factor (int): expansion factor of the MLP block
    """

    def __init__(
        self,
        dim: int = 512,
        layerscale: float = 0.1,
        expansion_factor: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim = dim
        self.layerscale = layerscale
        self.expansion_factor = expansion_factor

        # Affine Layers
        self.pre_affine = Affine(dim=self.dim)
        self.post_affine = Affine(dim=self.dim)

        # MLP Block
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.dim, out_features=self.expansion_factor * self.dim
            ),
            nn.GELU(),
            nn.Linear(
                in_features=self.expansion_factor * self.dim, out_features=self.dim
            ),
        )

        # LayerScale Parameter
        self.layerscale_val = nn.Parameter(
            data=torch.tensor(self.layerscale), requires_grad=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for CrossChannelSubLayer"""

        # Output from Affine Layer 1
        transform = self.pre_affine(inputs)

        # Feed into MLP Block
        mlp_output = self.mlp(transform)

        # Feed into Affine Layer 2
        affine_output = self.post_affine(mlp_output)

        # Skip-Connection with LayerScale
        output = inputs + affine_output * self.layerscale_val

        return output


class ResMLPLayer(nn.Module):
    """
    A PyTorch implementation of the ResMLP Layer

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        num_patches (int): no of patches
        dim (int): dimensionality for the Affine and MLP layers
        depth (int): number of blocks of the ResMLP Layer
        expansion_factor (int): expansion factor of the MLP block
    """

    def __init__(
        self,
        num_patches: int,
        dim: int = 512,
        depth: int = 12,
        expansion_factor: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.expansion_factor = expansion_factor

        # Determine value of LayerScale based on the depth
        if self.depth <= 18:
            self.layerscale = 0.1
        elif self.depth > 18 and self.depth <= 24:
            self.layerscale = 1e-5
        else:
            self.layerscale = 1e-6

        # Affine Layers
        self.patch_pre_affine = Affine(dim=self.dim)
        self.patch_post_affine = Affine(dim=self.dim)

        # Cross-Patch SubLayer
        self.crosspatchsublayer = nn.Sequential(
            self.patch_pre_affine,
            Rearrange("b n d -> b d n"),
            nn.Linear(self.num_patches, self.num_patches),
            Rearrange("b d n -> b n d"),
            self.patch_post_affine,
        )

        # Cross-Channel SubLayer
        self.crosschannelsublayer = CrossChannelSubLayer(
            dim=self.dim,
            layerscale=self.layerscale,
            expansion_factor=self.expansion_factor,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResMLPLayer"""

        # Feed into Cross-Patch SubLayer
        crosspatchsublayer_output = self.crosspatchsublayer(inputs)

        # Feed into Cross-Channel SubLayer
        crosschannelsublayer_output = self.crosschannelsublayer(
            crosspatchsublayer_output
        )

        return crosschannelsublayer_output


class ResMLP(nn.Module):
    """
    A PyTorch implementation of the ResMLP

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): dimensionality for the Affine and MLP layers
        depth (int): number of ResMLP layers
        in_channels (int): number of input channels
        num_classes (int): number of classes
        patch_size (int): size of the patches
        image_size (int): size of the image
        expansion_factor (int): expansion factor of the MLP block
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 12,
        in_channels: int = 3,
        num_classes: int = 10,
        patch_size: int = 16,
        image_size: int = 224,
        expansion_factor: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.patch_size = patch_size
        self.image_size = image_size
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.dim = dim
        self.depth = depth
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.expansion_factor = expansion_factor

        # Patch Projector
        self.patch_projector = nn.Sequential(
            Rearrange("b h w c-> b c h w"),  # channels last to channels first
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )

        # ResMLP Layers
        self.resmlp_layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.resmlp_layers.append(
                ResMLPLayer(
                    dim=self.dim,
                    depth=self.depth,
                    num_patches=self.num_patches,
                    expansion_factor=self.expansion_factor,
                )
            )

        # Classification Head
        self.classification_head = nn.Linear(self.dim, self.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResMLP"""

        # Get the Patch Embeddings
        patches = self.patch_projector(inputs)

        # Feed into ResMLP Layers
        for resmlp_layer in self.resmlp_layers:
            resmlp_output = resmlp_layer(patches)

        # Get the mean of the ResMLP Output
        mean = torch.mean(resmlp_output, dim=1)

        # Feed into Classification Head
        classification_head_output = self.classification_head(mean)

        return classification_head_output
