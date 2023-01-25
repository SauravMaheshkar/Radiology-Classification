"""Test Affine Tensorflow Layer"""

import numpy as np
import pytest

import tensorflow as tf
from src.tensorflow.modules.affine import Affine


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="dim",
    argvalues=[
        (512),
        (1024),
    ],
    ids=["dim-512", "dim-1024"],
)
def test_affine(dim: int) -> None:
    """Test Affine module"""
    temp_array = tf.ones(shape=(1, 1, dim))
    affine = Affine(dim=dim)
    affine.build(input_shape=temp_array.shape)

    assert np.allclose(
        affine.call(inputs=temp_array),
        np.ones(shape=(1, 1, dim)),
    )
