import jax
import jax.numpy as jnp
import numpy as np


# TODO: For non-evenly split grids this is slightly wrong since we essentially calculate the derivative at location
#   (t_(i-1) + t_(i+1)) / 2, not t_i.
def central_differences_first(
    x: np.ndarray | jax.Array, t: float | np.ndarray | jax.Array, keep_len: bool = True
) -> np.ndarray:
    if isinstance(x, jax.Array):
        xp = jnp
    else:
        xp = np
    if xp.isscalar(t):
        t = t * (xp.arange(x.shape[-2]) * xp.ones(x.shape[:-1])).astype(x.dtype)

    result = (x[:, 2:] - x[:, :-2]) / ((t[:, 2:] - t[:, :-2]) % 60.0)[..., None]
    if keep_len:
        nan_padding = np.nan * np.ones((x.shape[0], 1, *x.shape[2:]))
        return np.concatenate((nan_padding, result, nan_padding), axis=1)
    else:
        return result


def central_differences_second(
    x: np.ndarray | jax.Array, t: float | np.ndarray | jax.Array, keep_len: bool = True
) -> np.ndarray:
    if isinstance(x, jax.Array):
        xp = jnp
    else:
        xp = np
    if xp.isscalar(t):
        t = t * (xp.arange(x.shape[-2]) * xp.ones(x.shape[:-1])).astype(x.dtype)

    one_dim_input = x.ndim == 1
    if one_dim_input:
        x = x.reshape(-1, 1)
    v_next = (x[..., 2:, :] - x[..., 1:-1, :]) / ((t[..., 2:] - t[..., 1:-1]) % 60.0)[
        ..., None
    ]
    v_prev = (x[..., 1:-1, :] - x[..., :-2, :]) / ((t[..., 1:-1] - t[..., :-2]) % 60.0)[
        ..., None
    ]
    result = 2.0 * (v_next - v_prev) / ((t[..., 2:] - t[..., :-2]) % 60.0)[..., None]
    if keep_len:
        nan_padding = np.nan * np.ones((*x.shape[:-2], 1, x.shape[-1]))
        result = np.concatenate((nan_padding, result, nan_padding), axis=-2)
    if one_dim_input:
        result = result.squeeze(-1)
    return result


def backward_differences_first(
    x: np.ndarray | jax.Array, t: float | np.ndarray | jax.Array, keep_len: bool = True
) -> np.ndarray:
    if isinstance(x, jax.Array):
        xp = jnp
    else:
        xp = np
    if xp.isscalar(t):
        t = t * (xp.arange(x.shape[-2]) * xp.ones(x.shape[:-1])).astype(x.dtype)

    result = (x[..., 1:, :] - x[..., :-1, :]) / ((t[..., 1:] - t[..., :-1]) % 60.0)[
        ..., None
    ]
    if keep_len:
        nan_padding = np.nan * np.ones((*x.shape[:-2], 1, *x.shape[2:]))
        return np.concatenate((nan_padding, result), axis=-2)
    return result
