"""Test JAX utility functions"""

import numpy as np
import pytest

from jax import random
from src.jax.utils import full, ones, zeros


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed",
    argvalues=[0, 42],
    ids=["seed-0", "seed-42"],
)
def test_full(seed: int) -> None:
    """Test full function"""
    assert np.allclose(
        full(key=random.PRNGKey(seed), shape=(1, 1, 512), fill_value=1.0),
        np.ones(shape=(1, 1, 512)),
    )


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed",
    argvalues=[0, 42],
    ids=["seed-0", "seed-42"],
)
def test_ones(seed: int) -> None:
    """Test ones function"""
    assert np.allclose(
        ones(key=random.PRNGKey(seed), shape=(1, 1, 512)),
        np.ones(shape=(1, 1, 512)),
    )


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed",
    argvalues=[0, 42],
    ids=["seed-0", "seed-42"],
)
def test_zeros(seed: int) -> None:
    """Test zeros function"""
    assert np.allclose(
        zeros(key=random.PRNGKey(seed), shape=(1, 1, 512)),
        np.zeros(shape=(1, 1, 512)),
    )
