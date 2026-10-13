from typing import Optional, Sequence, Tuple

import numpy as np

from sim2real.data_processing.finite_differences import backward_differences_first


class TrajectoryDataset:
    def __init__(
        self,
        num_trajectories: int,
        trajectory_length: int,
    ):
        num_dofs = 4
        self.positions = np.nan * np.ones(
            (num_trajectories, trajectory_length + 1, num_dofs)
        )
        self.desired_pressures = np.nan * np.ones(
            (num_trajectories, trajectory_length + 1, num_dofs, 2), dtype=int
        )
        self.observed_pressures = np.nan * np.ones(
            (num_trajectories, trajectory_length + 1, num_dofs, 2), dtype=int
        )
        self.controls = np.nan * np.ones(
            (num_trajectories, trajectory_length, num_dofs)
        )
        self.timestamps = np.nan * np.ones((num_trajectories, trajectory_length + 1))

        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length

        self._traj_idx = 0

    @classmethod
    def from_data(
        cls,
        positions: np.ndarray,
        desired_pressures: Optional[np.ndarray],
        observed_pressures: Optional[np.ndarray],
        controls: np.ndarray,
        timestamps: np.ndarray,
    ) -> "TrajectoryDataset":
        dataset = cls(controls.shape[0], controls.shape[1])
        dataset.positions = positions
        dataset.controls = controls
        dataset.timestamps = timestamps
        dataset.desired_pressures = desired_pressures
        dataset.observed_pressures = observed_pressures
        dataset._traj_idx = controls.shape[0]
        return dataset

    def add_trajectory(
        self,
        positions: np.ndarray,
        desired_pressures: np.ndarray,
        observed_pressures: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        assert np.all(timestamps < 1e8), "Timestamps should be in seconds"
        self.positions[self._traj_idx, :, :] = positions
        self.desired_pressures[self._traj_idx, :, :] = desired_pressures
        self.observed_pressures[self._traj_idx, :, :] = observed_pressures
        self.controls[self._traj_idx, :, :] = controls
        self.timestamps[self._traj_idx, :] = timestamps
        self._traj_idx += 1

    def get_fd_velocity_trajectories(
        self, indices: Optional[Sequence[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if indices is not None:
            indices = np.array(indices)
        else:
            indices = range(self.positions.shape[0])

        velocities = backward_differences_first(
            self.positions[indices], self.timestamps[indices]
        )
        velocities = velocities[:, 1:]
        positions = self.positions[indices, 1:]
        controls = self.controls[indices, 1:]
        desired_pressures = (
            self.desired_pressures[indices, 1:]
            if self.desired_pressures is not None
            else None
        )
        observed_pressures = (
            self.observed_pressures[indices, 1:]
            if self.observed_pressures is not None
            else None
        )
        timestamps = self.timestamps[indices, 1:]
        return (
            positions,
            velocities,
            desired_pressures,
            observed_pressures,
            controls,
            timestamps,
        )

    def __getitem__(self, item):
        return TrajectoryDataset.from_data(
            self.positions[item],
            (
                self.desired_pressures[item]
                if self.desired_pressures is not None
                else None
            ),
            (
                self.observed_pressures[item]
                if self.observed_pressures is not None
                else None
            ),
            self.controls[item],
            self.timestamps[item],
        )

    # To be backwards-compatible with old datasets (do not contain pressures)
    def __setstate__(self, state):
        self.positions = state["positions"]
        self.desired_pressures = state.get("desired_pressures")
        self.observed_pressures = state.get("observed_pressures")
        self.controls = state["controls"]
        self.timestamps = state["timestamps"]
        self._traj_idx = state["_traj_idx"]
        self.num_trajectories = state.get("num_trajectories", self.controls.shape[0])
        self.trajectory_length = state.get("trajectory_length", self.controls.shape[1])
