from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.spatial.transform

from sim2real.rl_tasks.physics_backend.physics_backend import ObjData

RACKET_WIDTH = 0.015
BALL_RADIUS = 0.02
SIM_STEP = 0.002


def ball_pos_vel_rel_racket(
    obj_data: Dict[str, ObjData],
    last_ball_pos_rel: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    fill_in_pos = (
        last_ball_pos_rel
        if last_ball_pos_rel is not None
        else jnp.zeros_like(obj_data["ball"].pos)
    )
    ball_pos_rel, ball_vel_rel = mujoco_to_racket_frame(
        obj_data["racket"], obj_data["ball"].pos, obj_data["ball"].linvel
    )
    # If the ball position is NaN (no ball detection), return the last ball position (or zeros if there is no
    # last ball position).
    ball_pos_rel = jnp.where(
        jnp.any(jnp.isnan(obj_data["ball"].pos), axis=-1)[..., None],
        fill_in_pos,
        ball_pos_rel,
    )
    return ball_pos_rel, ball_vel_rel


def mujoco_to_racket_frame(
    racket_data: ObjData, pos: jax.Array, vel: Optional[jax.Array] = None
) -> Tuple[jax.Array, jax.Array]:
    racket_rot = jax.scipy.spatial.transform.Rotation.from_quat(racket_data.quat)
    pos_racket = racket_rot.apply(pos - racket_data.pos, inverse=True)
    vel_racket = (
        racket_rot.apply(vel - racket_data.linvel, inverse=True)
        if vel is not None
        else None
    )
    return pos_racket, vel_racket


def racket_to_mujoco_frame(
    racket_data: ObjData, pos: jax.Array, vel: Optional[jax.Array] = None
) -> Tuple[jax.Array, jax.Array]:
    racket_rot = jax.scipy.spatial.transform.Rotation.from_quat(racket_data.quat)
    racket_pos = racket_data.pos
    pos_mujoco = racket_rot.apply(pos) + racket_pos
    vel_mujoco = (
        (racket_rot.apply(vel) + racket_data.linvel) if vel is not None else None
    )
    return pos_mujoco, vel_mujoco


def ball_racket_contact(ball_pos_rel: jax.Array, tolerance: float) -> jax.Array:
    # The lengths of the two ellipse axes that approximate the racket surface
    racket_axis_x = 0.15
    racket_axis_y = 0.16
    # Distance to the edge that we enforce
    ball_rel_midpoint = ball_pos_rel - jnp.array([0.0, 0.105, 0.0])
    rx = racket_axis_x / 2
    ry = racket_axis_y / 2
    within_ellipse = (ball_rel_midpoint[..., 0] / rx) ** 2 + (
        ball_rel_midpoint[..., 1] / ry
    ) ** 2 <= 1.0
    z_close = (
        jnp.abs(ball_rel_midpoint[..., 2])
        <= BALL_RADIUS + 0.5 * RACKET_WIDTH + tolerance
    )
    return jnp.logical_and(within_ellipse, z_close)


def ball_racket_contact_model(
    prev_obj_data: Dict[str, ObjData],
) -> Tuple[jax.Array, jax.Array]:
    contact_epsilon = jnp.array([0.78, 0.78, 0.78])
    # TODO: Consider backhands
    racket_rot = jax.scipy.spatial.transform.Rotation.from_quat(
        prev_obj_data["racket"].quat
    )
    ball_pos_racket_frame, _ = mujoco_to_racket_frame(
        prev_obj_data["racket"], prev_obj_data["ball"].pos
    )
    # TODO: Should also take into account the effects of the racket angular velocity
    ball_vel_racket_frame = racket_rot.apply(prev_obj_data["ball"].linvel, inverse=True)
    racket_vel_racket_frame = racket_rot.apply(
        prev_obj_data["racket"].linvel, inverse=True
    )
    pos_racket_surface = 0.5 * RACKET_WIDTH + BALL_RADIUS
    # TODO: Could interpolate between prev and next velocities (solve equation)
    # Time in seconds when the contact happens
    delta_t_contact = jnp.clip(
        -(ball_pos_racket_frame[..., 2] - pos_racket_surface)
        / (ball_vel_racket_frame[..., 2] - racket_vel_racket_frame[..., 2]),
        0.0,
        SIM_STEP,
    )
    post_contact_ball_vel_racket_frame = ball_vel_racket_frame * contact_epsilon
    post_contact_ball_vel_racket_frame = post_contact_ball_vel_racket_frame.at[
        ..., 2
    ].set(
        -ball_vel_racket_frame[..., 2] * contact_epsilon[2]
        + (1.0 + contact_epsilon[2]) * racket_vel_racket_frame[..., 2]
    )
    post_contact_ball_vel = racket_rot.apply(post_contact_ball_vel_racket_frame)
    post_contact_ball_pos = (
        prev_obj_data["ball"].pos
        + delta_t_contact[..., None] * prev_obj_data["ball"].linvel
        + (SIM_STEP - delta_t_contact)[..., None] * post_contact_ball_vel
    )
    return post_contact_ball_pos, post_contact_ball_vel
