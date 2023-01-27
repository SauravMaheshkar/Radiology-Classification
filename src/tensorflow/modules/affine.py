"""Affine Transformation Layer in Tensorflow"""

import tensorflow as tf

__all__ = ["Affine"]


class Affine(tf.keras.layers.Layer):
    """
    A Tensorflow Keras Layer to perform a Affine Transformation

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): Needed to generate matrices of the appropriate shape
    """

    def __init__(self, dim: int = 512, **kwargs) -> None:
        super().__init__()

        # Attributes
        self.dim = dim

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the Affine layer based on the input shape"""

        # Generate the alpha and beta matrices
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1, 1, self.dim),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, self.dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Compute a forward pass through the Affine Transformation Layer"""
        return inputs * self.alpha + self.beta

    def get_config(self) -> dict:
        """Add dim to the config"""
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Return the output shape"""
        return input_shape
