"""Test Tensorflow ResMLP model"""

import pytest

import tensorflow as tf
from src.tensorflow.models.resmlp import (
    CrossChannelSubLayer,
    CrossPatchSubLayer,
    ResMLP,
    ResMLPLayer,
)

BATCH_SIZE = 4


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim, num_patches",
    argvalues=[
        (512, 16),
        (1024, 64),
    ],
    ids=["dim-512-patches-16", "dim-1024-patches-64"],
)
def test_crosspatchsublayer(dim: int, num_patches: int) -> None:
    """Test CrossPatchSubLayer module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, num_patches, dim))

    crosspatchsublayer = CrossPatchSubLayer(dim=dim, num_patches=num_patches)
    temp_output = crosspatchsublayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_patches, dim)


@pytest.mark.tensorflow
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

    temp_array = tf.ones(shape=(BATCH_SIZE, dim, dim))

    crosschannelsublayer = CrossChannelSubLayer(
        dim=dim, expansion_factor=expansion_factor
    )
    temp_output = crosschannelsublayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim, num_patches",
    argvalues=[
        (512, 8),
        (1024, 16),
    ],
    ids=["dim-512-patches-8", "dim-1024-patches-16"],
)
def test_resmlplayer(dim: int, num_patches: int) -> None:
    """Test ResMLPLayer module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, num_patches, dim))

    resmlplayer = ResMLPLayer(dim=dim, num_patches=num_patches, depth=2)
    temp_output = resmlplayer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_patches, dim)


@pytest.mark.tensorflow
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
    """Test ResMLP module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, image_size, image_size, in_channels))

    resmlp = ResMLP(
        dim=dim,
        depth=2,
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=image_size,
    )
    temp_output = resmlp(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
