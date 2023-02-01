"""Test Tensorflow FNet model"""

import pytest

import tensorflow as tf
from src.tensorflow.models.fnet import (
    FNet,
    FNetEncoderBlock,
    FNetFeedForwardLayer,
    FourierTransformLayer,
)

BATCH_SIZE = 4


@pytest.mark.tensorflow
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

    temp_array = tf.ones(shape=(BATCH_SIZE, dim, dim))

    fouriertransformlayer = FourierTransformLayer()
    temp_output = fouriertransformlayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.tensorflow
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

    temp_array = tf.ones(shape=(BATCH_SIZE, dim, dim))

    fnetfeedforwardlayer = FNetFeedForwardLayer()
    temp_output = fnetfeedforwardlayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.tensorflow
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

    temp_array = tf.ones(shape=(BATCH_SIZE, dim, dim))

    fnetencoderblock = FNetEncoderBlock(
        FourierTransformLayer(),
        FNetFeedForwardLayer(),
    )
    temp_output = fnetencoderblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim_ff, patch_size, num_classes, image_size, in_channels",
    argvalues=[
        (512, 16, 10, 224, 3),
        (512, 32, 10, 224, 3),
        (1024, 32, 10, 224, 3),
        (512, 16, 10, 224, 1),
    ],
)
def test_fnet(
    dim_ff: int, patch_size: int, num_classes: int, image_size: int, in_channels: int
) -> None:
    """Test FNet module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, image_size, image_size, in_channels))

    fnet = FNet(
        dim_ff=dim_ff,
        patch_size=patch_size,
        num_layers=2,
        num_classes=num_classes,
        image_size=image_size,
        dropout_rate=0.1,
    )
    temp_output = fnet(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
