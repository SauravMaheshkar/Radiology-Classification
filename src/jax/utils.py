"""JAX utility functions"""
from typing import Optional

import jax.numpy as jnp
from jax._src.typing import ArrayLike, DTypeLike, Shape

__all__ = ["full", "ones", "zeros"]


def full(
    key,  # pylint: disable=unused-argument
    shape: Shape,
    fill_value: ArrayLike,
    dtype: Optional[DTypeLike] = None,
) -> ArrayLike:
    """Helper function to create a full array of a given shape and dtype."""
    return jnp.full(shape, fill_value, dtype)


def ones(
    key,  # pylint: disable=unused-argument
    shape: Shape,
    dtype: Optional[DTypeLike] = None,
) -> ArrayLike:
    """Helper function to create an array of ones of a given shape and dtype."""
    return jnp.ones(shape, dtype)


def zeros(
    key,  # pylint: disable=unused-argument
    shape: Shape,
    dtype: Optional[DTypeLike] = None,
) -> ArrayLike:
    """Helper function to create an array of zeros of a given shape and dtype."""
    return jnp.zeros(shape, dtype)
