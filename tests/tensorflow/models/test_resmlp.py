"""Test Tensorflow ResMLP model"""

import pytest

import tensorflow as tf
from src.tensorflow.models.resmlp import (
    CrossChannelSubLayer,
    CrossPatchSubLayer,
    ResMLPLayer,
)


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim, patch_size",
    argvalues=[
        (512, 16),
        (1024, 64),
    ],
    ids=["dim-512-patch-16", "dim-1024-patch-64"],
)
def test_crosspatchsublayer(dim: int, patch_size: int) -> None:
    """Test CrossPatchSubLayer module"""

    temp_array = tf.ones(shape=(dim, dim))

    crosspatchsublayer = CrossPatchSubLayer(dim=dim, patch_size=patch_size)
    temp_output = crosspatchsublayer(temp_array)

    assert temp_output.shape == (patch_size, dim, dim)


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

    temp_array = tf.ones(shape=(dim, dim))

    crosschannelsublayer = CrossChannelSubLayer(
        dim=dim, expansion_factor=expansion_factor
    )
    temp_output = crosschannelsublayer(temp_array)

    assert temp_output.shape == (1, dim, dim)


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim, patch_size",
    argvalues=[
        (512, 8),
        (1024, 16),
    ],
    ids=["dim-512-patch-8", "dim-1024-patch-16"],
)
def test_resmlplayer(dim: int, patch_size: int) -> None:
    """Test ResMLPLayer module"""

    temp_array = tf.ones(shape=(dim, dim))

    resmlplayer = ResMLPLayer(dim=dim, patch_size=patch_size, depth=2)
    temp_output = resmlplayer(temp_array)

    assert temp_output.shape == (patch_size, dim, dim)
