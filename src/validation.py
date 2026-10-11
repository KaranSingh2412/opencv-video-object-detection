"""Shared validation so invalid inputs fail before reaching OpenCV."""

import math
from numbers import Integral, Real

import numpy as np


# --------------------------------------
# 1. Validate Input Image
# --------------------------------------

def validate_image(image, name="Image", *, grayscale=False):

    if image is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")

    if image.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have uint8 datatype")

    valid_shape = image.ndim == 2 or (
        not grayscale and image.ndim == 3 and image.shape[2] in (3, 4)
    )

    if not valid_shape:
        raise ValueError(f"{name} must be grayscale" if grayscale else f"{name} has unsupported image shape")


# --------------------------------------
# 2. Validate Integer Setting
# --------------------------------------

def validate_integer(value, name, *, minimum=1, maximum=None, odd=False):

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")

    if value < minimum or (maximum is not None and value > maximum):
        limit = f" between {minimum} and {maximum}" if maximum is not None else f" at least {minimum}"

        raise ValueError(f"{name} must be{limit}")

    if odd and value % 2 == 0:
        raise ValueError(f"{name} must be odd")


# --------------------------------------
# 3. Validate Nonnegative Setting
# --------------------------------------

def validate_nonnegative(value, name):

    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
