from jax.typing import ArrayLike


def standardize(x: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    standardized = (x - mean) / std
    return standardized


def de_standardize(x: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    de_standardized = (x * std) + mean
    return de_standardized
