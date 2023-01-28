"""Test Tensorflow MLPMixer model"""

import pytest

import tensorflow as tf
from src.tensorflow.models.mlpmixer import MixerBlock, MLPBlock, MLPMixer

BATCH_SIZE = 4


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="mlp_dim, num_channels, num_patches",
    argvalues=[
        (512, 16, 196),
        (1024, 32, 196),
    ],
    ids=[
        "mlp_dim-512-num_channels-16-num_patches-196",
        "mlp_dim-1024-num_channels-32-num_patches-196",
    ],
)
def test_mlpblock(mlp_dim: int, num_channels: int, num_patches: int) -> None:
    """Test MLPBlock module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, num_channels, num_patches))

    mlpblock = MLPBlock(mlp_dim=mlp_dim)
    temp_output = mlpblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_channels, num_patches)


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="tokens_mlp_dim, channels_mlp_dim, num_channels, num_patches",
    argvalues=[
        (384, 3072, 16, 196),
        (768, 6144, 32, 196),
    ],
    ids=[
        "tokens_mlp_dim-384-channels_mlp_dim-3072-num_channels-16-num_patches-196",
        "tokens_mlp_dim-768-channels_mlp_dim-6144-num_channels-32-num_patches-196",
    ],
)
def test_mixerblock(
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    num_channels: int,
    num_patches: int,
) -> None:
    """Test MixerBlock module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, num_channels, num_patches))

    mixerblock = MixerBlock(
        tokens_mlp_dim=tokens_mlp_dim, channels_mlp_dim=channels_mlp_dim
    )
    temp_output = mixerblock(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_channels, num_patches)


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    argnames="""
    image_size, in_channels, hidden_dim, patch_size, num_classes, tokens_mlp_dim, channels_mlp_dim
    """,
    argvalues=[
        (224, 3, 512, 16, 10, 384, 3072),
        (224, 1, 512, 16, 10, 384, 3072),
        (224, 3, 1024, 16, 10, 768, 6144),
    ],
)
def test_mlpmixer(
    image_size: int,
    in_channels: int,
    hidden_dim: int,
    patch_size: int,
    num_classes: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
) -> None:
    """Test MLPMixer module"""

    temp_array = tf.ones(shape=(BATCH_SIZE, image_size, image_size, in_channels))

    mlpmixer = MLPMixer(
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        num_blocks=2,
        num_classes=num_classes,
        tokens_mlp_dim=tokens_mlp_dim,
        channels_mlp_dim=channels_mlp_dim,
    )
    temp_output = mlpmixer(temp_array)

    assert temp_output.shape == (BATCH_SIZE, num_classes)
