import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import hydra
import mujoco
import numpy as np
import omegaconf
from matplotlib import pyplot as plt
from tqdm import tqdm

from sim2real.actuator_network.pack_an_input import pack_an_input
from sim2real.data_collection.trajectory_dataset import TrajectoryDataset
from sim2real.data_processing.finite_differences import (
    backward_differences_first,
    central_differences_second,
)
from sim2real.util.model import create_model_spec

logger = logging.getLogger(__name__)


def plot_dataset_visualizations(
    positions: np.ndarray,
    velocities: np.ndarray,
    controls: np.ndarray,
    torques: np.ndarray,
    outdir: Path,
) -> None:
    outdir.mkdir(exist_ok=True)
    for max_torque_dof in range(4):
        max_torque_idx_flat = np.argmax(torques[..., max_torque_dof])
        max_torque_idx = np.unravel_index(
            max_torque_idx_flat, torques[..., max_torque_dof].shape
        )
        fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
        for i, (y_label, values) in enumerate(
            [
                ("Position (°)", positions[max_torque_idx[0]] * 180.0 / np.pi),
                ("Velocities (°/s)", velocities[max_torque_idx[0]] * 180.0 / np.pi),
                ("Controls", controls[max_torque_idx[0]]),
                ("Torques (Nm)", torques[max_torque_idx[0]]),
            ]
        ):
            axes[i].axvline(
                max_torque_idx[1], -1e9, 1e9, color="grey", linestyle="dotted"
            )
            for dof in range(4):
                axes[i].plot(
                    values[:, dof],
                    label=f"Dof {dof}",
                    linestyle="solid" if dof == max_torque_dof else "dashed",
                )
            axes[i].set_xlabel("Timesteps")
            axes[i].set_ylabel(y_label)
        axes[-1].legend()
        fig.suptitle(
            f"Maximum torque trajectory for dof {max_torque_dof} (index: {max_torque_idx[0]})"
        )
        fig.tight_layout()
        # outdir_dof = outdir / f"max_torque_dof{max_torque_dof}"
        # outdir_dof.mkdir()
        fig.savefig(outdir / f"max_torque_trajectory_{max_torque_dof}.pdf")
        plt.close(fig)


def subsample_data(
    data: TrajectoryDataset, subsampling_factor: int
) -> TrajectoryDataset:
    assert data.controls.shape[1] % subsampling_factor == 0
    positions = data.positions[:, ::subsampling_factor]
    controls = data.controls[:, ::subsampling_factor]
    desired_pressures = (
        data.desired_pressures[:, ::subsampling_factor]
        if data.desired_pressures is not None
        else None
    )
    observed_pressures = (
        data.observed_pressures[:, ::subsampling_factor]
        if data.observed_pressures is not None
        else None
    )
    timestamps = data.timestamps[:, ::subsampling_factor]
    return TrajectoryDataset.from_data(
        positions,
        desired_pressures,
        observed_pressures,
        controls,
        timestamps,
    )


def apply_actuation_delay(
    data: TrajectoryDataset, actuation_delay: int, subsampling_factor: int
) -> TrajectoryDataset:
    # To simulate a delay of n steps, the instantaneous controls need to be shifted by n steps to the left
    controls = data.controls[:, actuation_delay:]
    desired_pressures = data.desired_pressures[:, actuation_delay:]

    controls = controls[:, actuation_delay % subsampling_factor :]
    desired_pressures = desired_pressures[:, actuation_delay % subsampling_factor :]
    positions = data.positions[:, actuation_delay % subsampling_factor :]
    observed_pressures = data.observed_pressures[
        :, actuation_delay % subsampling_factor :
    ]
    timestamps = data.timestamps[:, actuation_delay % subsampling_factor :]

    positions = positions[:, : controls.shape[1] + 1]
    observed_pressures = observed_pressures[:, : positions.shape[1]]
    timestamps = timestamps[:, : positions.shape[1]]

    return TrajectoryDataset.from_data(
        positions, desired_pressures, observed_pressures, controls, timestamps
    )


def compute_torques(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    end_effector: Literal["racket", "cup"],
) -> np.ndarray:
    assert positions.shape == velocities.shape == accelerations.shape
    assert end_effector in ["racket", "cup"], f"Invalid end effector: {end_effector}."
    if len(positions.shape) == 2:
        positions = positions[None]
        velocities = velocities[None]
        accelerations = accelerations[None]
    model = create_model_spec(end_effector).compile()
    model.opt.timestep = 0.002  # TODO: This should not matter --> Check!
    data = mujoco.MjData(model)
    torques = []
    for pos_traj, vel_traj, acc_traj in tqdm(
        zip(positions, velocities, accelerations),
        total=len(positions),
        desc="Computing torques",
    ):
        torques.append([])
        for pos, vel, acc in zip(pos_traj, vel_traj, acc_traj):
            data.qpos[:] = pos
            data.qvel[:] = vel
            data.qacc[:] = acc
            mujoco.mj_inverse(model, data)
            torques[-1].append(data.qfrc_inverse.copy())
    return np.array(torques)


def pad_history_length(
    positions: np.ndarray,
    positions_raw: Optional[np.ndarray],
    observed_pressures: Optional[np.ndarray],
    controls: np.ndarray,
    hist_len: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    pad_width_pos = (
        (0, 0),
        (hist_len - 1, 0),
        (0, 0),
    )
    positions = np.pad(positions, pad_width_pos, mode="edge")
    if positions_raw is not None:
        positions_raw = np.pad(
            positions_raw, pad_width_pos, mode="constant", constant_values=np.nan
        )
    else:
        positions_raw = None
    if observed_pressures is not None:
        pad_width_p = (
            (0, 0),
            (hist_len - 1, 0),
            (0, 0),
            (0, 0),
        )
        observed_pressures = np.pad(observed_pressures, pad_width_p, mode="edge")
    else:
        observed_pressures = None
    pad_width_ctrl = (
        (0, 0),
        (hist_len - 1, 0),
        (0, 0),
    )
    controls = np.pad(controls, pad_width_ctrl, mode="edge")
    return positions, positions_raw, observed_pressures, controls


def build_dataset(
    positions: np.ndarray,
    observed_pressures: np.ndarray,
    controls: np.ndarray,
    torques: np.ndarray,
    actuator_net_cfg: Dict[str, Any] | omegaconf.DictConfig,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    assert positions.shape == torques.shape
    assert controls.shape == (
        positions.shape[0],
        positions.shape[1] - 1,
        positions.shape[2],
    )

    hist_len = actuator_net_cfg.num_history_states * actuator_net_cfg.step_size + 1
    positions, _, observed_pressures, controls = pad_history_length(
        positions, None, observed_pressures, controls, hist_len
    )
    position_hists = np.lib.stride_tricks.sliding_window_view(
        positions[:, :-1], hist_len + 1, axis=1
    ).swapaxes(2, 3)
    if observed_pressures is not None:
        obs_pressures_hists = np.lib.stride_tricks.sliding_window_view(
            observed_pressures[:, :-1], hist_len + 1, axis=1
        ).swapaxes(2, 4)
    else:
        obs_pressures_hists = None
    controls_hists = np.lib.stride_tricks.sliding_window_view(
        controls[:, 1:], hist_len, axis=1
    ).swapaxes(2, 3)

    input_data = pack_an_input(
        position_hists,
        obs_pressures_hists,
        controls_hists,
        actuator_net_cfg,
        dt,
    )
    input_data = input_data.reshape(-1, input_data.shape[-1])
    torques = torques[:, 1:-1]
    torques = torques.reshape(-1, torques.shape[-1])
    assert input_data.shape[0] == torques.shape[0]
    return input_data, torques


def process_data(
    raw_data: TrajectoryDataset,
    end_effector: Literal["racket", "cup"],
    filter_cfg: Dict[str, Any] | omegaconf.DictConfig,
    target_frequency: float,
    actuation_delay: int,
    plot_visualizations: bool = False,
    precision: Literal["single", "double"] = "single",
    compute_torque_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    target_timestep = 1.0 / target_frequency

    recording_timestep = np.mean(
        (raw_data.timestamps[:, 1:] - raw_data.timestamps[:, :-1]) % 60.0
    )
    subsampling_factor = target_timestep / recording_timestep
    assert np.isclose(
        subsampling_factor, round(subsampling_factor), atol=1e-3
    ), f"target_timestep ({target_timestep}s) is not a multiple of recording_timestep ({recording_timestep}s)."
    subsampling_factor = round(subsampling_factor)

    data = apply_actuation_delay(raw_data, actuation_delay, subsampling_factor)

    filter = hydra.utils.instantiate(filter_cfg)
    (
        positions,
        positions_raw,
        desired_pressures,
        observed_pressures,
        controls,
        timestamps,
    ) = filter.process_an_training_data(
        data.positions,
        data.desired_pressures,
        data.observed_pressures,
        data.controls,
        data.timestamps,
    )
    data = TrajectoryDataset.from_data(
        positions,
        desired_pressures,
        observed_pressures,
        controls,
        timestamps,
    )

    data = subsample_data(data, subsampling_factor)
    positions_raw = positions_raw[:, ::subsampling_factor]

    # If NaNs from the actuation delay are still in the controls, truncate the data to include only valid controls.
    num_nans = np.sum(np.isnan(data.controls[0, :, 0]))
    if num_nans > 0:
        data = data[:, :-num_nans]
    assert not np.any(np.isnan(data.controls))

    velocities = backward_differences_first(data.positions, data.timestamps)
    accelerations = central_differences_second(data.positions, data.timestamps)

    output_dtype = np.float32 if precision == "single" else np.float64
    # The inverse-dynamics labels cost one mj_inverse call per sample, which is pure waste for
    # consumers that only roll the simulation out (e.g. the analytic muscle model sysid).
    if compute_torque_labels:
        torques = compute_torques(
            data.positions, velocities, accelerations, end_effector
        ).astype(output_dtype)
    else:
        torques = None
    positions = data.positions.astype(output_dtype)
    positions_raw = positions_raw.astype(output_dtype)
    velocities = velocities.astype(output_dtype)
    observed_pressures = data.observed_pressures.astype(output_dtype)
    controls = data.controls.astype(output_dtype)

    if plot_visualizations and torques is not None:
        plot_dataset_visualizations(
            positions,
            velocities,
            controls,
            torques,
            Path("dataset_visualizations"),
        )
    return (
        positions,
        positions_raw,
        observed_pressures,
        controls,
        torques,
    )
