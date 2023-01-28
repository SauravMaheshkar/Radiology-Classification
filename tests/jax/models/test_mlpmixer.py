"""Test MLP Mixer model Flax module"""

import pytest

from jax import random
from src.jax.models.mlpmixer import MixerBlock, MLPBlock, MLPMixer
from src.jax.utils import ones

BATCH_SIZE = 4


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed, mlp_dim, num_channels, num_patches",
    argvalues=[
        (0, 512, 16, 196),
        (42, 1024, 32, 196),
    ],
    ids=[
        "seed-0-mlp_dim-512-num_channels-16-num_patches-196",
        "seed-42-mlp_dim-1024-num_channels-32-num_patches-196",
    ],
)
def test_mlpblock(seed: int, mlp_dim: int, num_channels: int, num_patches: int) -> None:
    """Test MLPBlock module"""

    temp_array = ones(
        key=random.PRNGKey(seed), shape=(BATCH_SIZE, num_channels, num_patches)
    )

    mlpblock = MLPBlock(mlp_dim=mlp_dim)
    variables = mlpblock.init(random.PRNGKey(seed), x=temp_array)
    temp_output = mlpblock.apply(variables, x=temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_channels, num_patches)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed, tokens_mlp_dim, channels_mlp_dim, num_channels, num_patches",
    argvalues=[
        (0, 384, 3072, 16, 196),
        (42, 768, 6144, 32, 196),
    ],
)
def test_mixerblock(
    seed: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    num_channels: int,
    num_patches: int,
) -> None:
    """Test MixerBlock module"""

    temp_array = ones(
        key=random.PRNGKey(seed), shape=(BATCH_SIZE, num_channels, num_patches)
    )

    mixerblock = MixerBlock(
        tokens_mlp_dim=tokens_mlp_dim, channels_mlp_dim=channels_mlp_dim
    )
    variables = mixerblock.init(random.PRNGKey(seed), x=temp_array)
    temp_output = mixerblock.apply(variables, x=temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_channels, num_patches)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="""
    seed, image_size, patch_size, in_channels, num_classes, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim
    """,
    argvalues=[
        (0, 224, 16, 3, 10, 8, 512, 384, 3072),
        (0, 224, 16, 1, 10, 8, 512, 384, 3072),
        (42, 224, 16, 3, 10, 12, 512, 768, 6144),
    ],
)
def test_mlpmixer(
    seed: int,
    image_size: int,
    patch_size: int,
    in_channels: int,
    num_classes: int,
    num_blocks: int,
    hidden_dim: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
) -> None:
    """Test MLPMixer module"""

    temp_array = ones(
        key=random.PRNGKey(seed),
        shape=(BATCH_SIZE, image_size, image_size, in_channels),
    )

    mlpmixer = MLPMixer(
        patch_size=patch_size,
        num_classes=num_classes,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        tokens_mlp_dim=tokens_mlp_dim,
        channels_mlp_dim=channels_mlp_dim,
    )
    variables = mlpmixer.init(random.PRNGKey(seed), inputs=temp_array)
    temp_output = mlpmixer.apply(variables, inputs=temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
