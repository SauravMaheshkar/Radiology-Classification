"""Test FNet model Flax module"""

import pytest

from jax import random
from src.jax.models.fnet import (
    FNet,
    FNetEncoderBlock,
    FNetFeedForwardLayer,
    FourierTransformLayer,
)
from src.jax.utils import ones

BATCH_SIZE = 4


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed, dim",
    argvalues=[
        (0, 16),
        (42, 32),
    ],
    ids=[
        "seed-0-dim-16",
        "seed-42-dim-32",
    ],
)
def test_fouriertransformlayer(seed: int, dim: int) -> None:
    """Test FourierTransformLayer module"""

    temp_array = ones(key=random.PRNGKey(seed), shape=(BATCH_SIZE, dim, dim))

    fouriertransformlayer = FourierTransformLayer()
    variables = fouriertransformlayer.init(random.PRNGKey(seed), inputs=temp_array)
    temp_output = fouriertransformlayer.apply(variables, inputs=temp_array)

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed, dim",
    argvalues=[
        (0, 16),
        (42, 32),
    ],
    ids=[
        "seed-0-dim-16",
        "seed-42-dim-32",
    ],
)
def test_fnetfeedforwardlayer(seed: int, dim: int) -> None:
    """Test FNetFeedForwardLayer module"""

    seed = random.PRNGKey(seed)
    main_key, params_key, dropout_key = random.split(key=seed, num=3)

    temp_array = ones(key=main_key, shape=(BATCH_SIZE, dim, dim))

    fnetfeedforwardlayer = FNetFeedForwardLayer(dim_ff=512)
    variables = fnetfeedforwardlayer.init(
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )
    temp_output = fnetfeedforwardlayer.apply(
        variables=variables,
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed, dim",
    argvalues=[
        (0, 16),
        (42, 32),
    ],
    ids=[
        "seed-0-dim-16",
        "seed-42-dim-32",
    ],
)
def test_fnetencoderblock(seed: int, dim: int) -> None:
    """Test FNetEncoderBlock module"""

    seed = random.PRNGKey(seed)
    main_key, params_key, dropout_key = random.split(key=seed, num=3)

    temp_array = ones(key=main_key, shape=(BATCH_SIZE, dim, dim))

    fnetencoderblock = FNetEncoderBlock(
        FourierTransformLayer(), FNetFeedForwardLayer(512)
    )
    variables = fnetencoderblock.init(
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )
    temp_output = fnetencoderblock.apply(
        variables=variables,
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )

    assert temp_output.shape == (BATCH_SIZE, dim, dim)


@pytest.mark.jax
@pytest.mark.parametrize(
    argnames="seed, image_size, in_channels, num_classes, patch_size, dim_ff",
    argvalues=[
        (0, 224, 3, 10, 16, 512),
        (0, 224, 3, 10, 32, 512),
        (0, 224, 3, 10, 32, 1024),
        (0, 224, 1, 10, 16, 512),
        (42, 224, 3, 10, 16, 512),
    ],
)
def test_fnet(
    seed: int,
    image_size: int,
    in_channels: int,
    num_classes: int,
    patch_size: int,
    dim_ff: int,
) -> None:
    """Test FNet module"""

    seed = random.PRNGKey(seed)
    main_key, params_key, dropout_key = random.split(key=seed, num=3)

    temp_array = ones(
        key=main_key, shape=(BATCH_SIZE, image_size, image_size, in_channels)
    )

    fnet = FNet(
        dim_ff=dim_ff,
        patch_size=patch_size,
        num_layers=2,
        num_classes=num_classes,
        image_size=image_size,
        dropout_rate=0.1,
    )
    variables = fnet.init(
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )
    temp_output = fnet.apply(
        variables=variables,
        rngs={"params": params_key, "dropout": dropout_key},
        inputs=temp_array,
        training=False,
    )

    assert temp_output.shape == (BATCH_SIZE, num_classes)
