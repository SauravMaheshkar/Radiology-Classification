"""Test Affine Flax module"""

import numpy as np
import pytest

from jax import random
from src.jax.modules.affine import Affine
from src.jax.utils import ones


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="dim, seed",
    argvalues=[
        (512, 0),
        (1024, 42),
    ],
    ids=["dim-512-seed-0", "dim-1024-seed-42"],
)
def test_affine(dim: int, seed: int) -> None:
    """Test Affine module"""
    temp_array = ones(key=random.PRNGKey(seed), shape=(1, 1, dim))
    affine = Affine(dim=dim)
    variables = affine.init(random.PRNGKey(seed), x=temp_array)

    assert np.allclose(
        affine.apply(variables, x=temp_array),
        np.ones(shape=(1, 1, dim)),
    )
