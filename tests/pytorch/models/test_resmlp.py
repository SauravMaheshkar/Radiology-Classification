"""Test Pytorch ResMLP model"""

import pytest
import torch

from src.pytorch.models.resmlp import CrossChannelSubLayer, ResMLP, ResMLPLayer

BATCH_SIZE = 4


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="dim, expansion_factor",
    argvalues=[
        (512, 4),
        (1024, 8),
    ],
    ids=["dim-512-expansion-4", "dim-1024-expansion-8"],
)
def test_crosschannelsublayer(dim: int, expansion_factor: int) -> None:
    """Test CrossChannelSubLayer module"""

    temp_array = torch.ones(size=(BATCH_SIZE, dim, dim))

    crosschannelsublayer = CrossChannelSubLayer(
        dim=dim, expansion_factor=expansion_factor
    )
    temp_output = crosschannelsublayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="num_patches, dim",
    argvalues=[
        (16, 512),
        (64, 1024),
    ],
    ids=["num_patches-16-dim-512", "num_patches-64-dim-1024"],
)
def test_resmlplayer(num_patches: int, dim: int) -> None:
    """Test ResMLPLayer module"""

    temp_array = torch.ones(size=(BATCH_SIZE, num_patches, dim))

    resmlplayer = ResMLPLayer(num_patches=num_patches, dim=dim, depth=2)
    temp_output = resmlplayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_patches, dim)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="dim, patch_size, image_size, in_channels, num_classes",
    argvalues=[
        (512, 16, 224, 3, 10),
        (512, 8, 224, 3, 10),
        (512, 16, 224, 1, 10),
        (1024, 16, 224, 3, 10),
    ],
    ids=[
        "dim-512-patch_size-16-image_size-224-in_channels-3-num_classes-10",
        "dim-512-patch_size-8-image_size-224-in_channels-3-num_classes-10",
        "dim-512-patch_size-16-image_size-224-in_channels-1-num_classes-10",
        "dim-1024-patch_size-16-image_size-224-in_channels-3-num_classes-10",
    ],
)
def test_resmlp(
    dim: int, patch_size: int, image_size: int, in_channels: int, num_classes: int
) -> None:
    """Test ResMLP model"""

    temp_array = torch.ones(size=(BATCH_SIZE, image_size, image_size, in_channels))

    resmlp = ResMLP(
        dim=dim,
        depth=2,
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=image_size,
    )
    temp_output = resmlp(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
