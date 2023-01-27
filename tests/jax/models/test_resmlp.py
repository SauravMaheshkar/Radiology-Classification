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

BATCH_SIZE = 4


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="dim, seed, num_patches",
    argvalues=[
        (512, 0, 16),
        (1024, 42, 64),
    ],
    ids=["dim-512-seed-0-patches-16", "dim-1024-seed-42-patches-64"],
)
def test_crosspatchsublayer(dim: int, seed: int, num_patches: int) -> None:
    """Test CrossPatchSubLayer module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(BATCH_SIZE, num_patches, dim))

    crosspatchsublayer = CrossPatchSubLayer(dim=dim, num_patches=num_patches)
    variables = crosspatchsublayer.init(random.PRNGKey(seed), x=temp_array)
    temp_output = crosspatchsublayer.apply(variables, x=temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_patches, dim)


@pytest.mark.jax
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

    temp_array = ones(key=random.PRNGKey(seed), shape=(BATCH_SIZE, dim, dim))

    crosschannelsublayer = CrossChannelSubLayer(
        dim=dim, expansion_factor=expansion_factor
    )
    variables = crosschannelsublayer.init(random.PRNGKey(seed), x=temp_array)
    temp_output = crosschannelsublayer.apply(variables, x=temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="dim, seed, num_patches",
    argvalues=[
        (512, 0, 8),
        (1024, 42, 16),
    ],
    ids=["dim-512-seed-0-patches-8", "dim-1024-seed-42-patches-16"],
)
def test_resmlplayer(dim: int, seed: int, num_patches: int) -> None:
    """Test ResMLPLayer module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(BATCH_SIZE, num_patches, dim))

    resmlplayer = ResMLPLayer(dim=dim, num_patches=num_patches, depth=2)
    variables = resmlplayer.init(random.PRNGKey(seed), x=temp_array)
    temp_output = resmlplayer.apply(variables, x=temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_patches, dim)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="dim, seed, patch_size, image_size, in_channels, num_classes",
    argvalues=[
        (512, 0, 16, 224, 3, 10),
        (512, 0, 8, 224, 3, 10),
        (512, 0, 16, 224, 1, 10),
        (1024, 42, 16, 224, 3, 10),
    ],
    ids=[
        "dim-512-seed-0-patch_size-16-image_size-224-in_channels-3-num_classes-10",
        "dim-512-seed-0-patch_size-8-image_size-224-in_channels-3-num_classes-10",
        "dim-512-seed-0-patch_size-16-image_size-224-in_channels-1-num_classes-10",
        "dim-1024-seed-42-patch_size-16-image_size-224-in_channels-3-num_classes-10",
    ],
)
def test_resmlp(
    seed: int,
    dim: int,
    patch_size: int,
    image_size: int,
    in_channels: int,
    num_classes: int,
) -> None:
    """Test ResMLP module"""

    temp_array = ones(
        key=random.PRNGKey(seed),
        shape=(BATCH_SIZE, image_size, image_size, in_channels),
    )

    resmlp = ResMLP(
        dim=dim,
        depth=2,
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=image_size,
    )
    variables = resmlp.init(random.PRNGKey(seed), x=temp_array)
    temp_output = resmlp.apply(variables, x=temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
