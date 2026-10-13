from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import omegaconf
from jax.typing import ArrayLike

from sim2real.data_processing.finite_differences import backward_differences_first


def pack_an_input(
    positions: ArrayLike,
    pressures: Optional[ArrayLike],
    controls: ArrayLike,
    actuator_net_cfg: Dict[str, Any] | omegaconf.DictConfig,
    dt: float,
    pad_incomplete_hist: bool = False,
) -> ArrayLike:
    if isinstance(positions, np.ndarray):
        xp = np
    else:
        xp = jnp

    num_dofs = 4
    num_history_states = actuator_net_cfg["num_history_states"]
    step_size = actuator_net_cfg["step_size"]
    hist_len = num_history_states * step_size + 1
    assert pad_incomplete_hist or positions.shape[-2:] == (
        hist_len + 1,
        num_dofs,
    ), (
        f"Expected shape[-2:] of inputs to be {(hist_len + 1, num_dofs)}, "
        f"but got shape {tuple(positions.shape[-2:])}."
    )
    assert positions.shape[-2] > 1
    assert positions.shape == (
        *controls.shape[:-2],
        controls.shape[-2] + 1,
        controls.shape[-1],
    )

    if pad_incomplete_hist:
        positions, pressures, controls = _pad_to_hist_len(
            positions, pressures, controls, hist_len
        )

    pos_context = (
        positions[..., 1::step_size, :]
        if actuator_net_cfg.get("position_history", True)
        else positions[..., -1:, :]
    )
    velocities = backward_differences_first(positions, dt, keep_len=False)
    if actuator_net_cfg.get("velocity_input", True):
        vel_context = (
            velocities[..., ::step_size, :]
            if actuator_net_cfg.get("velocity_history", True)
            else velocities[..., -1:, :]
        )
    else:
        vel_context = None
    ctrl_context = (
        controls[..., ::step_size, :]
        if actuator_net_cfg.get("control_history", True)
        else controls[..., -1:, :]
    )
    pressure_context = (
        pressures[..., ::step_size, :, :]
        if actuator_net_cfg["pressure_input"]
        else None
    )

    if actuator_net_cfg.get("delta_history", False):
        pos_context = context_to_delta(pos_context)
        if vel_context is not None:
            vel_context = context_to_delta(vel_context)
        ctrl_context = context_to_delta(ctrl_context)
        if pressure_context is not None:
            pressure_context = context_to_delta_pressures(pressure_context)

    samples_shape = positions.shape[:-2]
    fields = [ctrl_context, pos_context]
    if vel_context is not None:
        fields.append(vel_context)
    if pressure_context is not None:
        fields.extend([pressure_context[..., 0], pressure_context[..., 1]])
    return xp.concatenate(fields, axis=-2).swapaxes(-1, -2).reshape(*samples_shape, -1)


def _pad_to_hist_len(
    positions: ArrayLike,
    pressures: Optional[ArrayLike],
    controls: ArrayLike,
    hist_len: int,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    if isinstance(positions, np.ndarray):
        xp = np
    else:
        xp = jnp

    first_pos_padding = positions[..., :1, :].repeat(
        repeats=hist_len - positions.shape[-2] + 1, axis=-2
    )
    positions = xp.concatenate((first_pos_padding, positions), axis=-2)
    first_ctrl_padding = controls[..., :1, :].repeat(
        repeats=hist_len - controls.shape[-2], axis=-2
    )
    controls = xp.concatenate((first_ctrl_padding, controls), axis=-2)
    if pressures is not None:
        first_pressure_padding = pressures[..., :1, :, :].repeat(
            repeats=hist_len - pressures.shape[-3],
            axis=-3,
        )
        pressures = xp.concatenate((first_pressure_padding, pressures), axis=-3)
    return positions, pressures, controls


def unpack_an_input(
    an_input: ArrayLike,
    actuator_net_cfg: Dict[str, Any] | omegaconf.DictConfig,
) -> Dict[str, ArrayLike]:
    context_length = actuator_net_cfg["num_history_states"] + 1
    # The *_history flags shrink the corresponding block to the newest entry only, mirroring
    # pack_an_input and get_an_input_size.
    ctrl_context_length = (
        context_length if actuator_net_cfg.get("control_history", True) else 1
    )
    pos_context_length = (
        context_length if actuator_net_cfg.get("position_history", True) else 1
    )
    vel_context_length = (
        context_length if actuator_net_cfg.get("velocity_history", True) else 1
    )
    size_per_dof = an_input.shape[-1] // 4
    an_input_transposed = an_input.reshape(*an_input.shape[:-1], 4, size_per_dof)

    curr_pos = 0
    ctrl_context = an_input_transposed[
        ..., curr_pos : curr_pos + ctrl_context_length
    ].swapaxes(-2, -1)
    curr_pos += ctrl_context_length
    pos_context = an_input_transposed[
        ..., curr_pos : curr_pos + pos_context_length
    ].swapaxes(-2, -1)
    curr_pos += pos_context_length
    if actuator_net_cfg["velocity_input"]:
        vel_context = an_input_transposed[
            ..., curr_pos : curr_pos + vel_context_length
        ].swapaxes(-2, -1)
        curr_pos += vel_context_length
    else:
        vel_context = None
    if actuator_net_cfg["pressure_input"]:
        pressure_context = (
            an_input_transposed[..., curr_pos : curr_pos + 2 * context_length]
            .reshape(*an_input_transposed.shape[:-1], 2, context_length)
            .swapaxes(-2, -1)
            .swapaxes(-3, -2)
        )
        curr_pos += 2 * context_length
    else:
        pressure_context = None

    if actuator_net_cfg["delta_history"]:
        pos_context = context_from_delta(pos_context)
        if vel_context is not None:
            vel_context = context_from_delta(vel_context)
        ctrl_context = context_from_delta(ctrl_context)
        if pressure_context is not None:
            pressure_context = context_from_delta_pressures(pressure_context)

    result = {
        "control_context": ctrl_context,
        "velocity_context": vel_context,
        "position_context": pos_context,
        "pressure_context": pressure_context,
    }

    # Output arrays shape: [..., time, dofs]
    return result


def context_to_delta(arr: ArrayLike) -> ArrayLike:
    if isinstance(arr, np.ndarray):
        xp = np
    else:
        xp = jnp
    return xp.concatenate(
        (
            arr[..., :-1, :] - arr[..., -1:, :],
            arr[..., -1:, :],
        ),
        axis=-2,
    )


def context_to_delta_pressures(arr: ArrayLike) -> ArrayLike:
    if isinstance(arr, np.ndarray):
        xp = np
    else:
        xp = jnp
    return xp.concatenate(
        (
            arr[..., :-1, :, :] - arr[..., -1:, :, :],
            arr[..., -1:, :, :],
        ),
        axis=-3,
    )


def context_from_delta(arr: ArrayLike) -> ArrayLike:
    if isinstance(arr, np.ndarray):
        xp = np
    else:
        xp = jnp
    return xp.concatenate(
        (
            arr[..., :-1, :] + arr[..., -1:, :],
            arr[..., -1:, :],
        ),
        axis=-2,
    )


def context_from_delta_pressures(arr: ArrayLike) -> ArrayLike:
    if isinstance(arr, np.ndarray):
        xp = np
    else:
        xp = jnp
    return xp.concatenate(
        (
            arr[..., :-1, :, :] + arr[..., -1:, :, :],
            arr[..., -1:, :, :],
        ),
        axis=-3,
    )
