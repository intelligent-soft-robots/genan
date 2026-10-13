from typing import Union

import jax.numpy as jnp
import numpy as np

ENCODER_TICKS = jnp.array([20000, 18400, 14400, 14400])
ENCODER_RESOLUTION = 2.0 * jnp.pi / ENCODER_TICKS
ENCODER_TICKS_NP = np.array(ENCODER_TICKS)
ENCODER_RESOLUTION_NP = np.array(ENCODER_RESOLUTION)


def encoder_quantization(
    positions: Union[np.ndarray, jnp.ndarray],
) -> Union[np.ndarray, jnp.ndarray]:
    if isinstance(positions, np.ndarray):
        simulated_encoder_ticks = np.expand_dims(
            ENCODER_TICKS_NP, tuple(range(positions.ndim - 1))
        )
    else:
        simulated_encoder_ticks = jnp.expand_dims(
            ENCODER_TICKS, tuple(range(positions.ndim - 1))
        )
    ticks = (positions * simulated_encoder_ticks / (2.0 * jnp.pi)).round()
    return ticks / simulated_encoder_ticks * 2.0 * jnp.pi
