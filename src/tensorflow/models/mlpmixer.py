"""Tensorflow implementation of MLP-Mixer"""

import tensorflow as tf


class MLPBlock(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of the MLPBlock

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        mlp_dim (int): no of dimensions for the MLP layers
    """

    def __init__(self, mlp_dim: int = 512, **kwargs) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.mlp_dim = mlp_dim

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for MLPBlock"""

        linear_output = tf.keras.layers.Dense(units=self.mlp_dim, activation="gelu")(
            inputs
        )
        return tf.keras.layers.Dense(units=inputs.shape[-1])(linear_output)

    def get_config(self) -> dict:
        """Get the config for the MLPBlock"""

        config = super().get_config()
        config.update({"mlp_dim": self.mlp_dim})
        return config


class MixerBlock(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of the MixerBlock consisting
    of a Token Mixing and Channel Mixing sublayer

    References:
        - https://arxiv.org/abs/2105.01601v1

    Attributes:
        tokens_mlp_dim (int): no of dimensions for the token mixing MLP layers
        channels_mlp_dim (int): no of dimensions for the channel mixing MLP layers
    """

    def __init__(
        self,
        tokens_mlp_dim: int = 384,
        channels_mlp_dim: int = 3072,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

        # Token Mixing
        self.token_mixing = MLPBlock(mlp_dim=self.tokens_mlp_dim)

        # Channel Mixing
        self.channel_mixing = MLPBlock(mlp_dim=self.channels_mlp_dim)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for MixerBlock"""

        # Token Mixing
        prenorm = tf.keras.layers.LayerNormalization()(inputs)
        prenorm_transposed = tf.transpose(prenorm, perm=(0, 2, 1))
        token_mixing_output = self.token_mixing(prenorm_transposed)

        # Channel Mixing
        token_mixing_output_transposed = tf.transpose(
            token_mixing_output, perm=(0, 2, 1)
        )
        skip_connection = inputs + token_mixing_output_transposed
        postnorm = tf.keras.layers.LayerNormalization()(skip_connection)

        return inputs + self.channel_mixing(postnorm)

    def get_config(self) -> dict:
        """Get the config for the MixerBlock"""

        config = super().get_config()
        config.update(
            {
                "tokens_mlp_dim": self.tokens_mlp_dim,
                "channels_mlp_dim": self.channels_mlp_dim,
            }
        )
        return config


class MLPMixer(tf.keras.Model):
    """A Tensorflow implementation of the MLP-Mixer architecture

    References:
        - https://arxiv.org/abs/2105.01601v1
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        patch_size: int = 16,
        num_blocks: int = 12,
        num_classes: int = 10,
        tokens_mlp_dim: int = 384,
        channels_mlp_dim: int = 3072,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        # Attributes
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

        # Patch Projector
        self.patch_projector = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.hidden_dim,
                    kernel_size=self.patch_size,
                    strides=self.patch_size,
                ),
                tf.keras.layers.Reshape(
                    target_shape=(
                        -1,
                        self.hidden_dim,
                    )
                ),
            ]
        )

        # Mixer Blocks
        self.mixer_blocks = [
            MixerBlock(
                tokens_mlp_dim=self.tokens_mlp_dim,
                channels_mlp_dim=self.channels_mlp_dim,
            )
            for _ in range(self.num_blocks)
        ]

        # Classifier
        self.head = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(units=self.num_classes),
            ]
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """Forward pass for MLPMixer"""

        # Patch Projector
        patch_projector_output = self.patch_projector(inputs)

        # Mixer Blocks
        mixer_blocks_output = patch_projector_output
        for mixer_block in self.mixer_blocks:
            mixer_blocks_output = mixer_block(mixer_blocks_output)

        # Classifier
        return self.head(mixer_blocks_output)

    def get_config(self) -> dict:
        """Returns the configuration of the model"""

        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "num_blocks": self.num_blocks,
                "num_classes": self.num_classes,
                "tokens_mlp_dim": self.tokens_mlp_dim,
                "channels_mlp_dim": self.channels_mlp_dim,
            }
        )
        return config
