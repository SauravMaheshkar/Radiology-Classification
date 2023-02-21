"""Test gMLP Flax module"""

import numpy as np
import pytest

from jax import random
from src.jax.modules.gmlp import SpatialGatingUnit, gMLPBlock
from src.jax.utils import ones

BATCH_SIZE = 4


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="dim, seed",
    argvalues=[
        (512, 0),
        (1024, 42),
    ],
    ids=["dim-512-seed-0", "dim-1024-seed-42"],
)
def test_spacialgatingunit(dim: int, seed: int) -> None:
    """Test SpatialGatingUnit module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(BATCH_SIZE, dim, dim))

    sgu = SpatialGatingUnit(dim=dim)
    variables = sgu.init(random.PRNGKey(seed), inputs=temp_array)
    temp_output = sgu.apply(variables, inputs=temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, (dim // 2))


@pytest.mark.jax
def test_gmlpblock() -> None:
    """Test gMLPBlock module"""

    temp_array = ones(key=random.PRNGKey(0), shape=(BATCH_SIZE, 512, 196))

    gmlpblock = gMLPBlock()
    variables = gmlpblock.init(random.PRNGKey(0), inputs=temp_array)
    temp_output = gmlpblock.apply(variables, inputs=temp_array)

    assert temp_output.shape == (BATCH_SIZE, 512, 196)
