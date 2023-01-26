"""Test Affine Pytorch Layer"""

import numpy as np
import pytest
import torch

from src.pytorch.modules.affine import Affine


@pytest.mark.pytorch
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
    temp_array = torch.ones(size=(1, 1, dim))
    affine = Affine(dim=dim)
    temp_output = affine(temp_array)

    assert np.allclose(
        temp_output.detach().numpy(),
        np.ones(shape=(1, 1, dim)),
    )
