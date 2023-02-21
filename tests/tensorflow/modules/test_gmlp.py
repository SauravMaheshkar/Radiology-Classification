"""Test Tensorflow gMLP model"""

import pytest

import tensorflow as tf
from src.tensorflow.modules.gmlp import SpatialGatingUnit, gMLPBlock

BATCH_SIZE = 4


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim",
    argvalues=[
        512,
        1024,
    ],
    ids=[
        "dim-512",
        "dim-1024",
    ],
)
def test_spacialgatingunit(dim: int) -> None:
    """Test SpatialGatingUnit module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, dim, dim))

    spatialgatingunit = SpatialGatingUnit(dim=dim)
    temp_output = spatialgatingunit(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, (dim // 2))


@pytest.mark.tensorflow
def test_gmlpblock() -> None:
    """Test gMLPBlock module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, 512, 196))

    gmlpblock = gMLPBlock()
    temp_output = gmlpblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, 512, 196)
