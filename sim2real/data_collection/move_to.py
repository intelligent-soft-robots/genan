import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import o80
import o80_pam_wrp
import yaml

from sim2real.data_collection.cocontraction import (
    Cocontraction,
)


class JointNotMovingException(Exception):
    pass


class JointOutOfBoundsException(Exception):
    pass


with (Path(__file__).parent / "config" / "joint_limits.yaml").open(
    "r"
) as joint_limits_path:
    joint_limits = yaml.safe_load(joint_limits_path)
    min_joint_angles = np.array(joint_limits["min"])
    max_joint_angles = np.array(joint_limits["max"])


def move_pressure_trajectory(
    pressures_traj: Sequence[Sequence[Sequence[int]]],  # (n_timesteps, n_joints, 2)
    frontend: o80_pam_wrp.FrontEnd,
    timestep: float,
    target_remain_time_s: Optional[float],
) -> List[o80_pam_wrp.Observation]:
    frequency = o80.FrequencyManager(1.0 / timestep)
    o80_time_step = o80.Duration_us.milliseconds(int(timestep * 1000))
    target_remain_time_iter = (
        round(target_remain_time_s / timestep)
        if target_remain_time_s is not None
        else 0
    )
    init_obs = frontend.latest()
    start_iteration = init_obs.get_iteration()
    for pressures in pressures_traj:
        for i_dof, (p_ago, p_antago) in enumerate(pressures):
            frontend.add_command(
                i_dof, p_ago, p_antago, o80_time_step, o80.Mode.OVERWRITE
            )
        frontend.pulse()
        frequency.wait()
    for _ in range(target_remain_time_iter):
        frequency.wait()
    return frontend.get_observations_since(start_iteration)


def get_control_trajectory(
    cocontraction,
    pressures_ago,
    pressures_antago,
    target_control,
    timestep: float,
    control_change_per_s: float = None,
    time_s: Optional[float] = None,
):
    start_control = cocontraction.pressures_to_control(pressures_ago, pressures_antago)
    target_control = np.asarray(target_control)

    assert time_s is not None or control_change_per_s is not None
    if time_s is not None:
        steps = math.ceil(time_s / timestep)
    else:
        control_change_per_iter = control_change_per_s * timestep
        steps = max(
            [
                math.ceil(abs(error) / control_change_per_iter)
                for error in target_control - start_control
            ]
        )

    return np.linspace(start_control, target_control, steps + 1)[1:]


def move_to_control(
    target_control: Sequence[float],
    frontend: o80_pam_wrp.FrontEnd,
    cocontraction: Cocontraction,
    timestep: float,
    control_change_per_s: Optional[float] = None,
    time_s: Optional[float] = None,
    target_remain_time_s: Optional[float] = None,
    skip_joint_moving_check: bool = False,
) -> List[o80_pam_wrp.Observation]:
    pressures_ago, pressures_antago = list(
        zip(*frontend.latest().get_observed_pressures())
    )
    control_trajectory = get_control_trajectory(
        cocontraction,
        pressures_ago,
        pressures_antago,
        target_control,
        timestep,
        control_change_per_s,
        time_s,
    )
    pressure_trajectory = np.stack(
        [
            np.stack(cocontraction.control_to_pressures(ctrl), axis=1)
            for ctrl in control_trajectory
        ],
        axis=0,
    )
    observations = move_pressure_trajectory(
        pressure_trajectory, frontend, timestep, target_remain_time_s
    )

    if not skip_joint_moving_check:
        # Sanity check if the robot is still moving if the control signals are changed (otherwise one of the tendons
        # might have ripped). If a tendon has ripped, the joint is usually
        for dof in range(4):
            init_ctrl = control_trajectory[0][dof]
            final_ctrl = control_trajectory[-1][dof]
            control_diff = final_ctrl - init_ctrl
            init_position = observations[0].get_positions()[dof]
            final_position = observations[-1].get_positions()[dof]
            pos_diff = final_position - init_position
            if (
                np.abs(control_diff) > 1.55
                and np.abs(pos_diff) < 0.05
                and final_position * 180 / np.pi - min_joint_angles[dof] < 20.0
                and max_joint_angles[dof] - final_position * 180 / np.pi < 20.0
            ):
                raise JointNotMovingException(
                    f"Joint {dof + 1} is not moving. Control difference: {control_diff}, "
                    f"position difference: {pos_diff}.\n"
                    f"Controls: {init_ctrl} --> {final_ctrl}, Positions: {init_position} --> {final_position}."
                )
        j4_final_position = observations[-1].get_positions()[3]
        if np.abs(j4_final_position * 180.0 / np.pi) > 200.0:
            raise JointOutOfBoundsException(
                f"J4 is out of bounds, found joint angle: {j4_final_position:.2f}°."
            )
    return observations


def move_to_pressures(
    target_pressures: Sequence[Sequence[int]],  # (n_joints, 2)
    frontend: o80_pam_wrp.FrontEnd,
    timestep: float,
    pressure_change_per_s: Optional[float] = None,
    time_s: Optional[float] = None,
    target_remain_time_s: Optional[float] = None,
):
    assert (
        pressure_change_per_s is None or time_s is None
    ), "Only one of pressure_change_per_s and time_s may be specified."

    current_pressures = np.array(frontend.latest().get_observed_pressures())
    target_pressures = np.asarray(target_pressures)
    assert target_pressures.shape == current_pressures.shape

    assert pressure_change_per_s is not None or time_s is not None
    if time_s is not None:
        steps = math.ceil(time_s / timestep)
    else:
        pressure_change_per_iter = pressure_change_per_s * timestep
        steps = max(
            [
                math.ceil(abs(error) / pressure_change_per_iter)
                for error in target_pressures - current_pressures
            ]
        )

    if steps == 0:
        return []
    else:
        pressure_trajectory = np.round(
            np.linspace(current_pressures, target_pressures, steps + 1)[1:]
        ).astype(int)

        return move_pressure_trajectory(
            pressure_trajectory, frontend, timestep, target_remain_time_s
        )
