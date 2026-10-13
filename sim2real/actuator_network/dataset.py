from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class Dataset:
    positions: np.ndarray | jnp.ndarray
    positions_raw: Optional[np.ndarray | jnp.ndarray]
    observed_pressures: np.ndarray | jnp.ndarray
    controls: np.ndarray | jnp.ndarray
    torques: Optional[np.ndarray | jnp.ndarray]

    def __getitem__(self, item):
        return Dataset(
            self.positions[item],
            self.positions_raw[item] if self.positions_raw is not None else None,
            self.observed_pressures[item],
            self.controls[item],
            self.torques[item] if self.torques is not None else None,
        )

    def to_jax(self) -> "Dataset":
        return Dataset(
            jnp.array(self.positions),
            jnp.array(self.positions_raw) if self.positions_raw is not None else None,
            jnp.array(self.observed_pressures),
            jnp.array(self.controls),
            jnp.array(self.torques) if self.torques is not None else None,
        )

    @property
    def num_trajectories(self) -> int:
        return self.positions.shape[0]

    @property
    def trajectory_length(self) -> int:
        return self.positions.shape[1]
