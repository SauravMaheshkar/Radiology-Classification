"""Tensorflow implementation of ResMLP."""

import tensorflow as tf

# local imports
from src.tensorflow.modules.affine import Affine

__all__ = ["CrossPatchSubLayer", "CrossChannelSubLayer", "ResMLPLayer", "ResMLP"]


class CrossPatchSubLayer(tf.keras.layers.Layer):
    """
    A Tensorflow implementation of the CrossPatchSubLayer

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): no of dimensions for the Affine and MLP layers
        num_patches (int): number of patches
        layerscale (float): value for the layerscale
    """

    def __init__(
        self, dim: int = 512, num_patches: int = 16, layerscale: float = 0.1, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim = dim
        self.num_patches = num_patches
        self.layerscale = layerscale

        # Affine Layers
        self.affine_1 = Affine(dim=self.dim)
        self.affine_2 = Affine(dim=self.dim)

        # Linear Layer
        self.linear = tf.keras.layers.Dense(units=self.num_patches)

        # LayerScale Parameter
        self.layerscale_val = tf.Variable(
            initial_value=self.layerscale, trainable=True, name="layerscale_crosspatch"
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for CrossPatchSubLayer"""

        # Output from Affine Layer 1
        transform = self.affine_1(inputs)

        # Transpose the Affine Layer 1
        transposed_transform = tf.transpose(transform, perm=(0, 2, 1))

        # Feed into Linear Layer
        linear_transform = self.linear(transposed_transform)

        # Tranpose the output from Linear Layer
        transposed_linear = tf.transpose(linear_transform, perm=(0, 2, 1))

        # Feed into Affine Layer 2
        affine_output = self.affine_2(transposed_linear)

        # Skip-Connection with LayerScale
        return inputs + affine_output * self.layerscale_val

    def get_config(self) -> dict:
        """Get the config for the CrossPatchSubLayer"""
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

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): no of dimensions for the Affine and MLP layers
        layerscale (float): value for the layerscale
        expansion_factor (int): expansion factor of the MLP block
    """

    def __init__(
        self,
        dim: int = 512,
        layerscale: float = 0.1,
        expansion_factor: int = 4,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.dim = dim
        self.layerscale = layerscale
        self.expansion_factor = expansion_factor

        # Affine Layers
        self.affine_1 = Affine(dim=self.dim)
        self.affine_2 = Affine(dim=self.dim)

        # MLP Block
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=self.expansion_factor * self.dim, activation="gelu"
                ),
                tf.keras.layers.Dense(units=self.dim),
            ]
        )

        # LayerScale Parameter
        self.layerscale_val = tf.Variable(
            initial_value=self.layerscale,
            trainable=True,
            name="layerscale_crosschannel",
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for CrossChannelSubLayer"""

        # Output from Affine Layer 1
        transform = self.affine_1(inputs)

        # Feed into MLP
        mlp_output = self.mlp(transform)

        # Feed into Affine Layer 2
        affine_output = self.affine_2(mlp_output)

        # Skip-Connection with LayerScale
        return inputs + affine_output * self.layerscale_val

    def get_config(self) -> dict:
        """Get the config for the CrossChannelSubLayer"""
        config = super().get_config()
        config.update({"dim": self.dim, "layerscale": self.layerscale})
        return config


class ResMLPLayer(tf.keras.layers.Layer):
    """A Tensorflow implementation of the ResMLP Layer.

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        num_patches (int): no of patches
        dim (int): dimensionality for the Affine and MLP layers
        depth (int): number of blocks of the ResMLP Layer
        expansion_factor (int): expansion factor of the MLP block
    """

    def __init__(
        self,
        num_patches: int,
        dim: int = 512,
        depth: int = 12,
        expansion_factor: int = 4,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.num_patches = num_patches
        self.expansion_factor = expansion_factor

        # Determine Value of LayerScale based on the depth
        if self.depth <= 18:
            self.layerscale = 0.1
        elif self.depth > 18 and self.depth <= 24:
            self.layerscale = 1e-5
        else:
            self.layerscale = 1e-6

        # Cross-Patch and Cross-Channel Sublayers
        self.crosspatch = CrossPatchSubLayer(
            dim=self.dim, num_patches=self.num_patches, layerscale=self.layerscale
        )
        self.crosschannel = CrossChannelSubLayer(
            dim=self.dim,
            layerscale=self.layerscale,
            expansion_factor=self.expansion_factor,
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Forward pass for ResMLPLayer"""

        # Cross-Patch Sublayer
        crosspatch_output = self.crosspatch(inputs)

        # Cross-Channel Sublayer
        crosschannel_output = self.crosschannel(crosspatch_output)

        return crosschannel_output

    def get_config(self) -> dict:
        """Get the config for the ResMLPLayer"""
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
    """A Tensorflow implementation of the ResMLP

    References:
        - https://arxiv.org/abs/2105.03404v2

    Attributes:
        dim (int): dimensionality for the Affine and MLP layers
        depth (int): number of ResMLP layers
        in_channels (int): number of input channels
        num_classes (int): number of classes
        patch_size (int): size of the patches
        image_size (int): size of the image
        expansion_factor (int): expansion factor of the MLP block
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 12,
        in_channels: int = 3,
        num_classes: int = 10,
        patch_size: int = 16,
        image_size: int = 224,
        expansion_factor: int = 4,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Attributes
        self.patch_size = patch_size
        self.image_size = image_size
        assert (
            self.image_size % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.dim = dim
        self.depth = depth
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.expansion_factor = expansion_factor

        # Patch Projector
        self.patch_projector = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.dim,
                    kernel_size=self.patch_size,
                    strides=self.patch_size,
                ),
                tf.keras.layers.Reshape(target_shape=(self.num_patches, self.dim)),
            ]
        )
        """self.patch_projector = tf.keras.layers.Conv2D(
            filters=self.dim, kernel_size=self.patch_size, strides=self.patch_size
        )"""

        # ResMLP Layers
        self.resmlp_layers = [
            ResMLPLayer(
                dim=self.dim,
                depth=self.depth,
                num_patches=self.num_patches,
                expansion_factor=self.expansion_factor,
            )
            for _ in range(self.depth)
        ]

        # Classification Head
        self.head = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(units=self.num_classes),
            ]
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """Forward pass for ResMLP"""

        # Get the Patch Embeddings
        patches = self.patch_projector(inputs)
        # patches = tf.reshape(patches, shape=(-1, self.num_patches, self.dim))

        # Feed into ResMLP Layers
        resmlp_output = patches
        for layer in self.resmlp_layers:
            resmlp_output = layer(resmlp_output)

        # Feed into Classification Head
        output = self.head(resmlp_output)

        return output

    def get_config(self) -> dict:
        """Get the config for the ResMLP"""
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
