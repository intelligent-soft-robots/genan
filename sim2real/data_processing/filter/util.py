from typing import Tuple

import numpy as np


def find_timestep(timestamps: np.ndarray) -> float:
    timestep = round(
        np.mean((timestamps[:, 1:] - timestamps[:, :-1]) % 60.0), ndigits=3
    )
    assert np.all(
        np.isclose((timestamps[:, 1:] - timestamps[:, :-1]) % 60.0, timestep, atol=5e-4)
    )
    return timestep


def interpolate(
    positions: np.ndarray, timestamps: np.ndarray, timestep: float
) -> Tuple[np.ndarray, np.ndarray]:
    pos_inter = []
    t_regular = np.arange(timestamps.shape[-1]) * timestep
    for t, pos in zip(timestamps, positions):
        pos_inter.append(
            np.stack(
                [
                    np.interp(t_regular, (t - t[:1]) % 60.0, p)
                    for p in np.moveaxis(pos, 0, -1)
                ],
                axis=-1,
            )
        )
    return np.stack(pos_inter, axis=0), t_regular[None].repeat(
        timestamps.shape[0], axis=0
    )
