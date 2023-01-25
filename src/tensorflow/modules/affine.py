"""Affine Transformation Layer in Tensorflow"""

import tensorflow as tf

__all__ = ["Affine"]


class Affine(tf.keras.layers.Layer):
    """
    A Tensorflow Keras Layer to perform a Affine Transformation

    Attributes:
        dim (int): Needed to generate matrices of the appropriate shape
    """

    def __init__(self, dim: int = 512, **kwargs) -> None:
        super(Affine, self).__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape: tf.TensorShape) -> None:
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
        return inputs * self.alpha + self.beta