"""Test PyTorch gMLP model"""

import pytest
import torch

from src.pytorch.modules.gmlp import SpatialGatingUnit, gMLPBlock

BATCH_SIZE = 4


@pytest.mark.pytorch
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

    temp_array = torch.ones(size=(BATCH_SIZE, dim, dim))

    spatialgatingunit = SpatialGatingUnit(dim=dim)
    temp_output = spatialgatingunit(temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, (dim // 2))


@pytest.mark.pytorch
def test_gmlpblock() -> None:
    """Test gMLPBlock module"""

    temp_array = torch.ones(size=(BATCH_SIZE, 512, 196))

    gmlpblock = gMLPBlock()
    temp_output = gmlpblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, 512, 196)
