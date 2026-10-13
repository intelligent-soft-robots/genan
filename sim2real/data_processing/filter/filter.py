import abc
from typing import Tuple

import numpy as np

from sim2real.data_processing.filter.util import find_timestep, interpolate


# TODO: Rename to preprocessor
class Filter(abc.ABC):
    def __init__(self, cutoff: int = 0, interpolate: bool = True):
        self.cutoff = cutoff
        self.interpolate = interpolate

    def process_an_training_data(
        self,
        positions: np.ndarray,
        desired_pressures: np.ndarray,
        observed_pressures: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.interpolate:
            timestep = find_timestep(timestamps)
            positions_interp, timestamps_interp = interpolate(
                positions, timestamps, timestep
            )
        else:
            positions_interp = positions
            timestamps_interp = timestamps
        positions_filtered = self._process_an_training_data(
            positions,
            positions_interp,
            timestamps,
            timestamps_interp,
        )
        if self.cutoff > 0:
            positions_filtered = positions_filtered[:, self.cutoff : -self.cutoff]
            positions_interp = positions_interp[:, self.cutoff : -self.cutoff]
            desired_pressures = (
                desired_pressures[:, self.cutoff : -self.cutoff]
                if desired_pressures is not None
                else None
            )
            observed_pressures = (
                observed_pressures[:, self.cutoff : -self.cutoff]
                if observed_pressures is not None
                else None
            )
            controls = controls[:, self.cutoff : -self.cutoff]
            timestamps_interp = timestamps_interp[:, self.cutoff : -self.cutoff]
        return (
            positions_filtered,
            positions_interp,
            desired_pressures,
            observed_pressures,
            controls,
            timestamps_interp,
        )

    @abc.abstractmethod
    def _process_an_training_data(
        self,
        positions: np.ndarray,
        positions_interp: np.ndarray,
        timestamps: np.ndarray,
        timestamps_interp: np.ndarray,
    ) -> np.ndarray:
        pass


class DefaultFilter(Filter):
    def _process_an_training_data(
        self,
        positions: np.ndarray,
        positions_interp: np.ndarray,
        timestamps: np.ndarray,
        timestamps_interp: np.ndarray,
    ) -> np.ndarray:
        return positions_interp
