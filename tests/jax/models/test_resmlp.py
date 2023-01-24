"""Test ResMLP model Flax module"""

import pytest

from jax import random
from src.jax.models.resmlp import (
    CrossChannelSubLayer,
    CrossPatchSubLayer,
    ResMLP,
    ResMLPLayer,
)
from src.jax.utils import ones


@pytest.mark.parametrize(
    argnames="dim, seed, patch_size",
    argvalues=[
        (512, 0, 16),
        (1024, 42, 64),
    ],
    ids=["dim-512-seed-0-patch-16", "dim-1024-seed-42-patch-64"],
)
def test_crosspatchsublayer(dim: int, seed: int, patch_size: int) -> None:
    """Test CrossPatchSubLayer module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(dim, dim))

    crosspatchsublayer = CrossPatchSubLayer(dim=dim, patch_size=patch_size)
    variables = crosspatchsublayer.init(random.PRNGKey(seed), x=temp_array)
    temp_output = crosspatchsublayer.apply(variables, x=temp_array)

    assert temp_output.shape == (patch_size, dim, dim)


@pytest.mark.parametrize(
    argnames="dim, seed, expansion_factor",
    argvalues=[
        (512, 0, 4),
        (1024, 42, 8),
    ],
    ids=["dim-512-seed-0-expansion-4", "dim-1024-seed-42-expansion-8"],
)
def test_crosschannelsublayer(dim: int, seed: int, expansion_factor: int) -> None:
    """Test CrossChannelSubLayer module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(dim, dim))

    crosschannelsublayer = CrossChannelSubLayer(
        dim=dim, expansion_factor=expansion_factor
    )
    variables = crosschannelsublayer.init(random.PRNGKey(seed), x=temp_array)
    temp_output = crosschannelsublayer.apply(variables, x=temp_array)

    assert temp_output.shape == (1, dim, dim)


@pytest.mark.parametrize(
    argnames="dim, seed, patch_size",
    argvalues=[
        (512, 0, 8),
        (1024, 42, 16),
    ],
    ids=["dim-512-seed-0-patch-8", "dim-1024-seed-42-patch-16"],
)
def test_resmlplayer(dim: int, seed: int, patch_size: int) -> None:
    """Test ResMLPLayer module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(dim, dim))

    resmlplayer = ResMLPLayer(dim=dim, patch_size=patch_size, depth=2)
    variables = resmlplayer.init(random.PRNGKey(seed), x=temp_array)
    temp_output = resmlplayer.apply(variables, x=temp_array)

    assert temp_output.shape == (patch_size, dim, dim)


@pytest.mark.parametrize(
    argnames="dim, seed, patch_size, num_classes",
    argvalues=[
        (512, 0, 8, 10),
        (1024, 42, 16, 100),
    ],
    ids=["dim-512-seed-0-patch-8-classes-10", "dim-1024-seed-42-patch-16-classes-100"],
)
def test_resmlp(seed: int, dim: int, patch_size: int, num_classes: int) -> None:
    """Test ResMLP module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(dim, dim))

    resmlp = ResMLP(dim=dim, patch_size=patch_size, num_classes=num_classes, depth=2)
    variables = resmlp.init(random.PRNGKey(seed), x=temp_array)
    temp_output = resmlp.apply(variables, x=temp_array)

    assert temp_output.shape == (patch_size, num_classes)
