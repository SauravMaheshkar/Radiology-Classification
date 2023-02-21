"""Tensorflow implementation of gMLP"""

import tensorflow as tf


class SpatialGatingUnit(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of the SpatialGatingUnit
    References:
        - https://arxiv.org/abs/2105.08050v2
    Attributes:
        dim (int): no of dimensions for the SpatialGatingUnit
    """

    def __init__(self, dim: int, name: str = "Spatial Gating Unit", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

        # Attributes
        self.dim = dim

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for SpatialGatingUnit"""
        # pylint: disable=invalid-name
        u, v = tf.split(inputs, num_or_size_splits=2, axis=-1)
        v = tf.keras.layers.LayerNormalization(epsilon=1e-6)(v)
        v = tf.linalg.matrix_transpose(v)
        v = tf.keras.layers.Dense(self.dim, bias_initializer="ones")(v)
        v = tf.linalg.matrix_transpose(v)

        return u * v
        # pylint: enable=invalid-name

    def get_config(self) -> dict:
        """Get the config for the SpatialGatingUnit"""

        config = super().get_config()
        config.update({"dim": self.dim})
        return config


class gMLPBlock(tf.keras.layers.Layer):  # pylint: disable=invalid-name
    """
    A Tensorflow implementation of the gMLPBlock
    References:
        - https://arxiv.org/abs/2105.08050v2
    """

    def __init__(self, name: str = "gMLP Block", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the gMLPBlock"""

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.in_proj = tf.keras.layers.Dense(input_shape[-1] * 2, activation="gelu")
        self.sgu = SpatialGatingUnit(dim=input_shape[-2])
        self.out_proj = tf.keras.layers.Dense(input_shape[-1])

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for gMLPBlock"""
        shortcut = inputs
        norm = self.norm(inputs)
        in_proj = self.in_proj(norm)
        sgu = self.sgu(in_proj)
        out_proj = self.out_proj(sgu)
        return out_proj + shortcut
