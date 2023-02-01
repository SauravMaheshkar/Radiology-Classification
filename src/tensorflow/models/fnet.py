"""Tensorflow implementaion of the FNet"""

import tensorflow as tf


class FourierTransformLayer(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of the FourierTransformLayer

    References:
        - https://arxiv.org/abs/2105.03824
    """

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for FourierTransformLayer"""
