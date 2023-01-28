"""Test PyTorch MLPMixer model"""

import pytest
import torch

from src.pytorch.models.mlpmixer import MixerBlock, MLPBlock, MLPMixer

BATCH_SIZE = 4


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="mlp_dim, num_channels, num_patches",
    argvalues=[
        (512, 16, 196),
        (1024, 32, 196),
    ],
    ids=[
        "mlp_dim-512-num_channels-16-num_patches-196",
        "mlp_dim-1024-num_channels-32-num_patches-196",
    ],
)
def test_mlpblock(mlp_dim: int, num_channels: int, num_patches: int) -> None:
    """Test MLPBlock module"""

    temp_array = torch.ones(size=(BATCH_SIZE, num_channels, num_patches))

    mlpblock = MLPBlock(mlp_dim=mlp_dim)
    temp_output = mlpblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_channels, num_patches)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="tokens_mlp_dim, channels_mlp_dim, num_channels, num_patches",
    argvalues=[
        (384, 3072, 16, 196),
        (768, 6144, 32, 196),
    ],
    ids=[
        "tokens_mlp_dim-384-channels_mlp_dim-3072-num_channels-16-num_patches-196",
        "tokens_mlp_dim-768-channels_mlp_dim-6144-num_channels-32-num_patches-196",
    ],
)
def test_mixerblock(
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    num_channels: int,
    num_patches: int,
) -> None:
    """Test MixerBlock module"""

    temp_array = torch.ones(size=(BATCH_SIZE, num_channels, num_patches))

    mixerblock = MixerBlock(
        tokens_mlp_dim=tokens_mlp_dim, channels_mlp_dim=channels_mlp_dim
    )
    temp_output = mixerblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_channels, num_patches)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="""
    in_channels, image_size, patch_size, hidden_dim, num_classes, tokens_mlp_dim, channels_mlp_dim
    """,
    argvalues=[
        (3, 224, 16, 512, 10, 384, 3072),
        (1, 224, 16, 512, 10, 384, 3072),
        (3, 224, 16, 512, 10, 768, 6144),
    ],
)
def test_mlpmixer(
    in_channels: int,
    image_size: int,
    patch_size: int,
    hidden_dim: int,
    num_classes: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
) -> None:
    """Test MLPMixer module"""

    temp_array = torch.ones(size=(BATCH_SIZE, image_size, image_size, in_channels))

    mlpmixer = MLPMixer(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_blocks=2,
        tokens_mlp_dim=tokens_mlp_dim,
        channels_mlp_dim=channels_mlp_dim,
    )
    temp_output = mlpmixer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
