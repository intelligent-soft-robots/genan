import json
import logging
import os
import pickle
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import o80
import o80_pam
import omegaconf
import pam_interface
import pam_mujoco
from matplotlib import pyplot as plt
from tqdm import tqdm

from sim2real.data_collection.cocontraction import (
    Cocontraction,
)
from sim2real.data_collection.trajectory_dataset import TrajectoryDataset
from sim2real.util.encoder_simulation import encoder_quantization

logger = logging.getLogger(__name__)


def init_robot(cfg: omegaconf.DictConfig):
    if cfg.device == "robot":
        frontend = o80_pam.FrontEnd("real_robot")
        robot_config = pam_interface.Pamy2DefaultConfiguration(False)
        mujoco_handle = None
        exp_robot_config = {
            "min_pressures_ago": [13000, 16500, 12500, 11800],
            "min_pressures_antago": [13000, 19000, 12200, 11800],
            "max_pressures_ago": [23000, 18500, 23500, 19000],
            "max_pressures_antago": [23000, 24500, 22500, 19000],
        }
    elif cfg.device == "simulator":
        mujoco_id = "data_collection"
        robot_segment_id = "robot"
        robot = pam_mujoco.MujocoRobot(
            pam_mujoco.RobotType.PAMY2,
            robot_segment_id,
            control=pam_mujoco.MujocoRobot.PRESSURE_CONTROL,
        )
        mujoco_handle = pam_mujoco.MujocoHandle(
            mujoco_id,
            robot1=robot,
            graphics=False,
            accelerated_time=True,
            burst_mode=True,
        )

        frontend = mujoco_handle.frontends[robot_segment_id]
        robot_config = pam_interface.JsonConfiguration(str(robot.json_control_path))
        exp_robot_config = {
            "min_pressures_ago": [15000, 15000, 13000, 13000],
            "min_pressures_antago": [16000, 16000, 13000, 13000],
            "max_pressures_ago": [25000, 21500, 21000, 19000],
            "max_pressures_antago": [25000, 25000, 21000, 21900],
        }
    else:
        raise ValueError(f"Unknown device {cfg.device}.")

    return frontend, robot_config, mujoco_handle


def log_workspace_commits():
    src_path = Path(os.environ["ROS_PACKAGE_PATH"]).parents[2] / "src"
    for repo_path in src_path.iterdir():
        if repo_path.is_dir() and (repo_path / ".git").exists():
            commit_sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path)
                .decode()
                .strip()
            )
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path
                )
                .decode()
                .strip()
            )
            logger.info(f"{repo_path.name}: {branch} ({commit_sha})")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="collect_data",
)
def collect(cfg: omegaconf.DictConfig):
    log_workspace_commits()
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    assert (
        cfg.overwrite or not Path("raw_data.pkl").exists()
    ), f"Dataset {Path.cwd() / 'raw_data.pkl'} already exists."

    frontend, robot_config, mujoco_handle = init_robot(cfg)
    with Path("robot_config.json").open("w") as f:
        robot_config_dict = {
            "min_pressures_ago": robot_config.min_pressures_ago,
            "min_pressures_antago": robot_config.min_pressures_antago,
            "max_pressures_ago": robot_config.max_pressures_ago,
            "max_pressures_antago": robot_config.max_pressures_antago,
            "control_period": robot_config.control_period,
            "sensor_period": robot_config.sensor_period,
        }
        json.dump(robot_config_dict, f, indent=4)

    motion_generator = hydra.utils.instantiate(
        cfg.motion_generator,
        backend_timestep=1.0 / frontend.get_frequency(),
    )
    delta_limit = cfg.delta_limit * np.ones(4)
    cocontraction = Cocontraction.from_robot_config(robot_config, cfg.ndp)
    logger.info("Started collecting data.")

    num_samples = round(
        cfg.motion_generator.duration / motion_generator.backend_timestep
    )
    data = TrajectoryDataset(
        cfg.num_trajectories,
        num_samples,
    )

    obs_pressures = np.array(frontend.latest().get_observed_pressures())
    last_controls = cocontraction.pressures_to_control(
        obs_pressures[:, 0], obs_pressures[:, 1]
    )
    last_pos_traj = None

    min_joint_angles = []
    max_joint_angles = []

    init_pressures_ago, init_pressures_antago = cocontraction.control_to_pressures(
        np.zeros(4)
    )
    frontend.add_command(
        init_pressures_ago,
        init_pressures_antago,
        o80.Duration_us.seconds(1),
        o80.Mode.OVERWRITE,
    )
    frontend.pulse()
    time.sleep(1.0)

    i = 0
    pbar = tqdm(total=cfg.num_trajectories)
    while i < cfg.num_trajectories:
        control_trajectory = motion_generator.generate_signal(
            last_controls, last_pos_traj
        )
        pressure_ago, pressure_antago = cocontraction.control_to_pressures(
            control_trajectory[0]
        )
        frontend.add_command(
            pressure_ago,
            pressure_antago,
            o80.Mode.OVERWRITE,
        )
        frontend.pulse()
        # Wait a second before actually starting to execute the commands to give the system time to stabilize.
        time.sleep(1.0)
        last_control = control_trajectory[0]
        max_velocity = np.array(cfg.max_velocity_deg_s) * np.pi / 180.0
        velocity_limit_violations = np.zeros(4, dtype=bool)
        initial_iteration = frontend.latest().get_iteration()
        curr_iteration = initial_iteration
        while True:
            while frontend.latest().get_iteration() == curr_iteration:
                if mujoco_handle is not None:
                    mujoco_handle.burst()
            obs = frontend.latest()
            curr_iteration = obs.get_iteration()
            idx = curr_iteration - initial_iteration
            if idx >= len(control_trajectory):
                break
            # If the velocity is larger than max_velocity, apply only half the delta control if it is accelerating
            # in the same direction (to slow down the movement).
            velocity = np.array(obs.get_velocities())
            velocity_limit_violations = np.logical_or(
                velocity_limit_violations, np.abs(velocity) > max_velocity
            )
            # Disable velocity_limit_violations only if the velocity is less than 80% of the maximum.
            velocity_limit_violations = np.logical_and(
                velocity_limit_violations, np.abs(velocity) > 0.8 * max_velocity
            )
            delta_control = control_trajectory[idx] - last_control
            delta_control = np.clip(delta_control, -delta_limit, delta_limit)
            delta_control = np.where(
                np.logical_and(
                    velocity_limit_violations,
                    np.sign(delta_control) == np.sign(velocity),
                ),
                np.clip(
                    0.1 * delta_control + np.random.normal(scale=0.002, size=4),
                    -delta_limit,
                    delta_limit,
                ),
                delta_control,
            )
            control = last_control + delta_control
            pressure_ago, pressure_antago = cocontraction.control_to_pressures(
                control
            )
            frontend.add_command(
                pressure_ago,
                pressure_antago,
                o80.Mode.OVERWRITE,
            )
            frontend.pulse()
            last_control = control

        while frontend.latest().get_iteration() == curr_iteration:
            if mujoco_handle is not None:
                mujoco_handle.burst()
        observations = frontend.get_observations_since(initial_iteration + 1)[
            : len(control_trajectory) + 1
        ]

        desired_pressures = np.array(
            [o.get_desired_pressures() for o in observations]
        )
        observed_pressures = np.array(
            [o.get_observed_pressures() for o in observations]
        )

        # If an iteration was skipped, the previous control was applied instead of the one specified
        # control_trajectory. Extract the actual controls for each iteration from the pressures.
        controls = cocontraction.pressures_to_control(
            desired_pressures[:-1, :, 0], desired_pressures[:-1, :, 1]
        )
        # Record the timestamp of the PLC (observation.get_extended_state().get_time_stamp()) instead of the host
        # timestamp (observation.get_time_stamp()).
        timestamps = np.array(
            [o.get_extended_state().get_time_stamp() for o in observations]
        ) * (1e-6 if cfg.device == "robot" else 1e-9)
        positions = np.array([o.get_positions() for o in observations])
        velocities = np.array([o.get_velocities() for o in observations])

        last_controls = controls[-1]
        last_pos_traj = positions

        if not np.all(
            np.isclose(
                (timestamps[1:] - timestamps[:-1]) % 60.0,
                motion_generator.backend_timestep,
                rtol=0,
                atol=5e-4,
            )
        ):
            max_step = np.max((timestamps[1:] - timestamps[:-1]) % 60.0)
            min_step = np.min((timestamps[1:] - timestamps[:-1]) % 60.0)
            if np.abs(max_step - motion_generator.backend_timestep) > np.abs(
                min_step - motion_generator.backend_timestep
            ):
                max_step_deviation = max_step
            else:
                max_step_deviation = min_step
            logger.warning(
                f"Expected timestamp differences to be {motion_generator.backend_timestep:.3f}, "
                f"got {max_step_deviation:.3f}. Skipping trajectory."
            )
        elif np.any(np.isnan(positions)) or np.any(np.isnan(velocities)):
            logger.warning("Encountered NaN values.")
            if mujoco_handle is not None:
                # For some reason resetting sometimes fails or takes longer than one step, just retry in that case
                while (
                    frontend.latest().get_extended_state().get_time_stamp()
                    > 2000000
                ):
                    mujoco_handle.reset()
                    mujoco_handle.burst()  # TODO: Remove this if the reset is fixed in pam_mujoco
                    logger.warning(
                        f"Resetting... (timestamp after reset: "
                        f"{frontend.latest().get_extended_state().get_time_stamp()})"
                    )
        elif mujoco_handle is not None and np.any(np.abs(positions) > np.pi / 2):
            idx_violation = np.argmax(np.max(np.abs(positions), axis=1))
            with np.printoptions(precision=2, suppress=True):
                logger.warning(
                    f"Joint limit violation (got angles: {positions[idx_violation] * 180.0 / np.pi}°)."
                )
            # For some reason resetting sometimes fails or takes longer than one step, just retry in that case
            while frontend.latest().get_extended_state().get_time_stamp() > 2000000:
                mujoco_handle.reset()
                mujoco_handle.burst()  # TODO: Remove this if the reset is fixed in pam_mujoco
                logger.warning(
                    f"Resetting... (timestamp after reset: "
                    f"{frontend.latest().get_extended_state().get_time_stamp()})"
                )
        elif np.any(np.abs(positions[1:] - positions[:-1]) > 0.1):
            logger.warning(
                f"Difference between consecutive joint angles is too large. Skipping trajectory."
            )
        # All checks passed --> Add the trajectory to the dataset
        else:
            if cfg.simulate_encoder_quantization:
                positions = encoder_quantization(positions)
            data.add_trajectory(
                positions,
                desired_pressures,
                observed_pressures,
                controls,
                timestamps,
            )

            min_joint_angles.append(np.min(positions, axis=0))
            max_joint_angles.append(np.max(positions, axis=0))

            with np.printoptions(precision=2, suppress=True):
                logger.info(
                    f"Averages: minimum joint angles: {np.mean(min_joint_angles, axis=0) * 180 / np.pi}°, "
                    f"maximum joint angles: {np.mean(max_joint_angles, axis=0) * 180 / np.pi}°"
                )
                logger.info(
                    f"Extremes: minimum joint angles: {np.min(min_joint_angles, axis=0) * 180 / np.pi}°, "
                    f"maximum joint angles: {np.max(max_joint_angles, axis=0) * 180 / np.pi}°"
                )

            i += 1
            pbar.update(1)
    pbar.close()

    frontend.add_command(
        cocontraction.min_pressures_ago_np,
        cocontraction.min_pressures_antago_np,
        o80.Duration_us.seconds(2),
        o80.Mode.OVERWRITE,
    )
    frontend.pulse()
    logger.info("Saving data.")
    with Path("raw_data.pkl").open("wb") as f:
        pickle.dump(data, f)

    num_episodes_to_plot = min(10, cfg.num_trajectories)
    colors_dofs = ["r", "g", "b", "y"]
    for i in range(num_episodes_to_plot):
        steps = np.arange(data.controls.shape[1]) + i * data.positions.shape[1]
        for j in range(4):
            plt.plot(
                steps,
                data.controls[i, :, j].reshape(-1),
                color=colors_dofs[j],
                label=f"$u_{j}$" if i == 0 else "_nolegend_",
            )
    plt.xlabel("t")
    plt.ylabel("u")
    plt.legend()
    plt.savefig("data_collection_commands.pdf")
    plt.close()

    for i in range(num_episodes_to_plot):
        steps = np.arange(data.positions.shape[1]) + i * data.positions.shape[1]
        for j in range(4):
            plt.plot(
                steps,
                data.positions[i, :, j] * 180.0 / np.pi,
                color=colors_dofs[j],
                label=f"$q_{j}$" if i == 0 else "_nolegend_",
            )
    plt.xlabel("t")
    plt.ylabel("Joint positions (°)")
    plt.legend()
    plt.savefig("data_collection_positions.pdf")
    plt.close()

    _, fd_velocities, _, _, _, fd_timestamps = data.get_fd_velocity_trajectories(
        np.arange(num_episodes_to_plot)
    )

    logger.info("Finished collecting data.")


if __name__ == "__main__":
    collect()
