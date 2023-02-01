"""Tensorflow implementaion of the FNet"""

import numpy as np

import tensorflow as tf

__all__ = ["FourierTransformLayer", "FNetFeedForwardLayer", "FNetEncoderBlock"]


class FourierTransformLayer(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of the FourierTransformLayer

    References:
        - https://arxiv.org/abs/2105.03824
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._fourier_transform = np.fft.fftn

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for FourierTransformLayer"""

        return self._fourier_transform(inputs)


class FNetFeedForwardLayer(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of a Feed Forward Layer
    for the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        dim_ff: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate

    def call(
        self, inputs: tf.Tensor, *args, training: bool = False, **kwargs
    ) -> tf.Tensor:
        """Forward pass for FNetFeedForwardLayer"""

        intermediate_dense_output = tf.keras.layers.Dense(
            units=self.dim_ff, activation="gelu"
        )(inputs)
        dense_output = tf.keras.layers.Dense(units=inputs.shape[-1])(
            intermediate_dense_output
        )
        output = tf.keras.layers.Dropout(rate=self.dropout_rate, trainable=training)(
            dense_output
        )
        return output

    def get_config(self) -> dict:
        """Get the config for the FNetFeedForwardLayer"""

        config = super().get_config()
        config.update(
            {
                "dim_ff": self.dim_ff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class FNetEncoderBlock(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of an Encoder Block
    for the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824
    """

    def __init__(
        self,
        fourier_layer: FourierTransformLayer,
        feed_forward_layer: FNetFeedForwardLayer,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.fourier_layer = fourier_layer
        self.feed_forward_layer = feed_forward_layer

    def call(
        self, inputs: tf.Tensor, *args, training: bool = False, **kwargs
    ) -> tf.Tensor:
        """Forward pass for FNetEncoderBlock"""

        mixing_output = self.fourier_layer(inputs)
        mixing_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)(
            mixing_output + inputs
        )
        feed_forward_output = self.feed_forward_layer(
            mixing_layer_norm, training=training
        )
        return tf.keras.layers.LayerNormalization(epsilon=1e-12)(
            feed_forward_output + mixing_layer_norm
        )

    def get_config(self) -> dict:
        """Get the config for the FNetEncoderBlock"""

        config = super().get_config()
        config.update(
            {
                "fourier_layer": self.fourier_layer,
                "feed_forward_layer": self.feed_forward_layer,
            }
        )
        return config


class FNet(tf.keras.Model):
    """
    A Tensorflow implementation of the FNet architecture

    References:
        - https://arxiv.org/abs/2105.03824

    Attributes:
        dim_ff (int): no of dimensions for the Feed Forward Layer
        patch_size (int): patch size
        num_layers (int): no of layers
        num_classes (int): no of classes
        image_size (int): image size
        dropout_rate (float): dropout rate
    """

    def __init__(
        self,
        dim_ff: int = 512,
        patch_size: int = 16,
        num_layers: int = 12,
        num_classes: int = 10,
        image_size: int = 224,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim_ff = dim_ff
        self.patch_size = patch_size
        self.image_size = image_size
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Patch Projector
        self.patch_projector = tf.keras.layers.Conv2D(
            filters=self.dim_ff,
            kernel_size=self.patch_size,
            strides=self.patch_size,
        )

        # Encoder Blocks
        self.encoder_blocks = [
            FNetEncoderBlock(
                fourier_layer=FourierTransformLayer(),
                feed_forward_layer=FNetFeedForwardLayer(
                    dim_ff=self.dim_ff, dropout_rate=self.dropout_rate
                ),
            )
            for _ in range(self.num_layers)
        ]

        # Classifier
        self.head = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(units=self.num_classes),
            ]
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        mask=None,
    ) -> tf.Tensor:
        """Forward pass for FNet"""

        # Patch Projector
        patch_projector_output = self.patch_projector(inputs)

        # Encoder Blocks
        encoder_blocks_output = patch_projector_output
        for encoder_block in self.encoder_blocks:
            encoder_blocks_output = encoder_block(
                encoder_blocks_output, training=training
            )

        # Classifier
        return self.head(encoder_blocks_output)

    def get_config(self) -> dict:
        """Get the config for the FNet"""

        config = super().get_config()
        config.update(
            {
                "dim_ff": self.dim_ff,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
