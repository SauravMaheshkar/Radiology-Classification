"""Test GeLU activation function module"""

import numpy as np
import pytest

from jax import random
from src.jax.activations.gelu import GeLU


@pytest.mark.jax
def test_gelu() -> None:
    """Test GeLU activation function"""
    temp_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    gelu = GeLU()
    variables = gelu.init(random.PRNGKey(0), x=temp_array)

    assert np.allclose(
        gelu.apply(variables, x=temp_array),
        np.array([0.0, 0.841192, 1.9545977, 2.9963627, 3.99993, 4.9999995]),
    )
