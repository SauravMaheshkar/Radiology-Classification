"""Test PyTorch FNet model"""

import pytest
import torch

from src.pytorch.models.fnet import (
    FNet,
    FNetEncoderBlock,
    FNetFeedForwardLayer,
    FourierTransformLayer,
)

BATCH_SIZE = 4


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="dim",
    argvalues=[
        (16),
        (32),
    ],
    ids=[
        "dim-16",
        "dim-32",
    ],
)
def test_fouriertransformlayer(dim: int) -> None:
    """Test FourierTransformLayer module"""

    temp_array = torch.ones(size=(BATCH_SIZE, dim, dim))

    fourier_transform_layer = FourierTransformLayer()
    temp_output = fourier_transform_layer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="dim",
    argvalues=[
        (16),
        (32),
    ],
    ids=[
        "dim-16",
        "dim-32",
    ],
)
def test_fnetfeedforwardlayer(dim: int) -> None:
    """Test FNetFeedForwardLayer module"""

    temp_array = torch.ones(size=(BATCH_SIZE, dim, dim))

    fnet_feed_forward_layer = FNetFeedForwardLayer()
    temp_output = fnet_feed_forward_layer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="dim",
    argvalues=[
        (16),
        (32),
    ],
    ids=[
        "dim-16",
        "dim-32",
    ],
)
def test_fnetencoderblock(dim: int) -> None:
    """Test FNetEncoderBlock module"""

    temp_array = torch.ones(size=(BATCH_SIZE, dim, dim))
    fnet_encoder_block = FNetEncoderBlock()
    temp_output = fnet_encoder_block(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.pytorch
@pytest.mark.parametrize(
    argnames="dim_ff, patch_size, image_size, in_channels",
    argvalues=[
        (512, 16, 224, 3),
        (512, 32, 224, 3),
        (1024, 32, 224, 3),
        (512, 16, 224, 1),
    ],
)
def test_fnet(dim_ff: int, patch_size: int, image_size: int, in_channels: int) -> None:
    """Test FNet module"""

    fnet = FNet(
        dim_ff=dim_ff,
        num_layers=2,
        num_classes=10,
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
    )
    temp_array = torch.ones(size=(BATCH_SIZE, image_size, image_size, in_channels))
    temp_output = fnet(temp_array)

    assert temp_output.shape == (BATCH_SIZE, 10)
