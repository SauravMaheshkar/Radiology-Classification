"""Tensorflow implementation of ResMLP."""

import tensorflow as tf

# local imports
from src.tensorflow.modules.affine import Affine

__all__ = ["CrossPatchSubLayer", "CrossChannelSubLayer", "ResMLPLayer"]


class CrossPatchSubLayer(tf.keras.layers.Layer):
    """A Tensorflow implementation of the CrossPatchSubLayer.

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        patch_size: dimensionality for the Linear Layer
        layerscale: value for the layerscale
    """

    def __init__(
        self, dim: int = 512, patch_size: int = 16, layerscale: float = 0.1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.layerscale = layerscale

        self.affine_1 = Affine(dim=self.dim)
        self.linear = tf.keras.layers.Dense(units=self.patch_size)
        self.affine_2 = Affine(dim=self.dim)
        self.layerscale_val = tf.Variable(
            initial_value=self.layerscale, trainable=True, name="layerscale_crosspatch"
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        # Output from Affine Layer 1
        transform = self.affine_1(inputs)

        # Transpose the Affine Layer 1
        transposed_transform = tf.transpose(transform)

        # Feed into Linear Layer
        linear_transform = self.linear(transposed_transform)

        # Tranpose the output from Linear Layer
        transposed_linear = tf.transpose(linear_transform)

        # Feed into Affine Layer 2
        affine_output = self.affine_2(transposed_linear)

        # Skip-Connection with LayerScale
        output = inputs + affine_output * self.layerscale_val

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "patch_size": self.patch_size,
                "layerscale": self.layerscale,
            }
        )
        return config


class CrossChannelSubLayer(tf.keras.layers.Layer):
    """A Tensorflow implementation of the CrossChannelSubLayer.

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        layerscale: value for the layerscale
        expansion_factor: expansion factor for the MLP
    """

    def __init__(
        self, dim: int = 512, layerscale: float = 0.1, expansion_factor: int = 4
    ) -> None:
        super().__init__()
        self.dim = dim
        self.layerscale = layerscale
        self.expansion_factor = expansion_factor

        self.affine_1 = Affine(dim=self.dim)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self.expansion_factor * self.dim, activation="gelu"
                ),
                tf.keras.layers.Dense(units=self.dim),
            ]
        )
        self.affine_2 = Affine(dim=self.dim)
        self.layerscale_val = tf.Variable(
            initial_value=self.layerscale,
            trainable=True,
            name="layerscale_crosschannel",
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        # Output from Affine Layer 1
        transform = self.affine_1(inputs)

        # Feed into MLP
        mlp_output = self.mlp(transform)

        # Feed into Affine Layer 2
        affine_output = self.affine_2(mlp_output)

        # Skip-Connection with LayerScale
        output = inputs + affine_output * self.layerscale_val

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"dim": self.dim, "layerscale": self.layerscale})
        return config


class ResMLPLayer(tf.keras.layers.Layer):
    """A Tensorflow implementation of the ResMLP Layer.

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        depth: No of ResMLP Layers, needed for determining the layerscale value
        patch_size: dimensionality for the Linear Layer
    """

    def __init__(self, dim: int = 512, depth: int = 12, patch_size: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size

        # Determine Value of LayerScale based on the depth
        if self.depth <= 18:
            self.layerscale = 0.1
        elif self.depth > 18 and self.depth <= 24:
            self.layerscale = 1e-5
        else:
            self.layerscale = 1e-6

        self.crosspatch = CrossPatchSubLayer(
            dim=self.dim, patch_size=self.patch_size, layerscale=self.layerscale
        )
        self.crosschannel = CrossChannelSubLayer(
            dim=self.dim, layerscale=self.layerscale
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        # Cross-Patch Sublayer
        crosspatch_output = self.crosspatch(inputs)

        # Cross-Channel Sublayer
        crosschannel_output = self.crosschannel(crosspatch_output)

        return crosschannel_output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "patch_size": self.patch_size,
                "layerscale": self.layerscale,
            }
        )
        return config


class ResMLP(tf.keras.Model):
    """A Tensorflow implementation of the ResMLP.

    Attributes:
        dim: dimensionality for the Affine and MLP layers
        depth: No of ResMLP Layers, needed for determining the layerscale value
        patch_size: dimensionality for the Linear Layer
        num_classes: number of classes for the classification head
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 12,
        patch_size: int = 16,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.patch_projector = tf.keras.layers.Conv2D(
            filters=self.dim, kernel_size=self.patch_size, strides=self.patch_size
        )

        self.resmlp_layers = [
            ResMLPLayer(dim=self.dim, depth=self.depth, patch_size=self.patch_size)
            for _ in range(self.depth)
        ]

        self.head = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(units=self.num_classes),
            ]
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        # Stem
        patches = self.patch_projector(inputs)

        # ResMLP Layers
        resmlp_output = patches
        for layer in self.resmlp_layers:
            resmlp_output = layer(resmlp_output)

        # Classification Head
        output = self.head(resmlp_output)

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "num_classes": self.num_classes,
            }
        )
        return config
