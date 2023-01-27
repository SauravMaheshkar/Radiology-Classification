"""Flax Module implementation of GeLU activation function"""

import numpy as np
from flax import linen as nn

import jax.numpy as jnp
from jax import lax
from jax._src.typing import ArrayLike

__all__ = ["GeLU"]


class GeLU(nn.Module):
    """
    A Flax linen Module to perform a GeLU activation function

    Attributes:
        approximate (bool): bool value to use approximate version of GeLU
    """

    approximate: bool = True

    @nn.compact
    def __call__(self, x, *args, **kwargs) -> ArrayLike:
        """Compute a forward pass through the GeLU activation function"""
        if self.approximate:
            sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
            cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
            return x * cdf
        else:
            return jnp.array(x * (lax.erf(x / np.sqrt(2)) + 1) / 2, dtype=x.dtype)
