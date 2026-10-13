from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class EKFState:
    def __init__(
        self,
        t: jax.Array,
        x: jax.Array,
        P: jax.Array,
        last_measurement: jax.Array,
    ):
        self.t = t
        self.x = x
        self.P = P
        self.last_measurement = last_measurement

    def tree_flatten(self):
        children = (
            self.t,
            self.x,
            self.P,
            self.last_measurement,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class BallEKF:
    def __init__(
        self,
        measurement_noise: float = 1e-3,
        transition_noise_pos: float = 1e-4,
        transition_noise_vel: float = 1e-2,
        transition_noise_spin: float = 1e-4,
        vel_diff_reset_threshold: float = 5.5,
    ):
        self.covR = jnp.eye(3) * measurement_noise
        self.covQ = jnp.diag(
            jnp.array(
                [transition_noise_pos] * 3
                + [transition_noise_vel] * 3
                + [transition_noise_spin] * 3
            )
        )
        self.vel_diff_reset_threshold = vel_diff_reset_threshold

        # Pre-build jacobians as functions (still traceable when used under jit)
        self._ball_model_jac = jax.vmap(jax.jacfwd(self._ball_model), in_axes=(0, 0))
        self._observation_model_jac = jax.vmap(jax.jacfwd(self._observation_model))

    def reset(self, pos: jax.Array, vel: jax.Array, timestamp: jax.Array) -> EKFState:
        x = jnp.concatenate((pos, vel, jnp.zeros_like(pos)), axis=-1)
        P = jnp.diag(jnp.array([1e-4] * 3 + [2.0] * 3 + [1e-2] * 3))[None].repeat(
            x.shape[0], axis=0
        )
        return EKFState(t=timestamp, x=x, P=P, last_measurement=pos)

    def predict(self, state: EKFState, t_query: jax.Array) -> EKFState:
        dt = t_query - state.t
        dt = jnp.maximum(dt, 0.0)

        A = self._ball_model_jac(state.x, dt)
        x_pred = self._ball_model(state.x, dt)
        P_pred = (
            jnp.matmul(jnp.matmul(A, state.P), jnp.swapaxes(A, -2, -1))
            + self.covQ[None]
        )

        return EKFState(
            t=t_query, x=x_pred, P=P_pred, last_measurement=state.last_measurement
        )

    def update(
        self, state: EKFState, pos_measurement: jax.Array, timestamp: jax.Array
    ) -> EKFState:
        state_pred = self.predict(state, timestamp)

        H = self._observation_model_jac(state_pred.x)
        S = jnp.matmul(jnp.matmul(H, state_pred.P), jnp.swapaxes(H, -2, -1)) + self.covR

        PHt = jnp.matmul(state_pred.P, jnp.swapaxes(H, -2, -1))
        K = jnp.linalg.solve(S, jnp.swapaxes(PHt, -2, -1))
        K = jnp.swapaxes(K, -2, -1)

        innov = pos_measurement - self._observation_model(state_pred.x)
        x_post = state_pred.x + jnp.matmul(K, innov[..., None]).squeeze(-1)
        P_post = state_pred.P - jnp.matmul(jnp.matmul(K, H), state_pred.P)
        state_post = EKFState(
            t=timestamp, x=x_post, P=P_post, last_measurement=pos_measurement
        )

        last_vel = state.x[..., 3:6]
        fd_vel = (pos_measurement - state.last_measurement) / (timestamp - state.t)[
            ..., None
        ]
        state_reset = self.reset(pos_measurement, fd_vel, timestamp)
        needs_reset = (
            jnp.linalg.norm(fd_vel - last_vel, axis=-1) > self.vel_diff_reset_threshold
        )
        # Check if the velocities changed abruptly --> reset
        next_state = jax.tree.map(
            lambda sp, sr: jnp.where(
                needs_reset.reshape((-1,) + (1,) * (sr.ndim - 1)), sr, sp
            ),
            state_post,
            state_reset,
        )
        # Check if either the measurement or the timestamp are masked (nan) --> keep last state
        nan_measurement = jnp.logical_or(
            jnp.any(jnp.isnan(pos_measurement), axis=-1), jnp.isnan(timestamp)
        )[..., None]
        next_state = jax.tree.map(
            lambda s, next_s: jnp.where(
                nan_measurement.reshape((-1,) + (1,) * (s.ndim - 1)), s, next_s
            ),
            state,
            next_state,
        )
        return next_state

    def filter_trajectory(
        self,
        pos_measurements: jax.Array,
        timestamps: jax.Array,
        timestamps_new_rel: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        reset_jit = jax.jit(self.reset)
        update_jit = jax.jit(self.update)
        timestamps = timestamps - timestamps[..., 1:2]
        states = [
            reset_jit(
                pos_measurements[..., 1, :],
                (pos_measurements[..., 1, :] - pos_measurements[..., 0, :])
                / (timestamps[..., 1] - timestamps[..., 0])[..., None],
                timestamps[..., 1],
            )
        ]
        for pos_meas, timestamp in zip(
            jnp.swapaxes(pos_measurements[..., 2:, :], 0, 1),
            jnp.swapaxes(timestamps[..., 2:], 0, 1),
        ):
            states.append(update_jit(states[-1], pos_meas, timestamp))
        states = jax.tree.map(lambda *s: jnp.stack(s, axis=1), *states)

        @jax.jit
        def get_prediction(t_new: jax.Array) -> EKFState:
            t_too_large = states.t > t_new
            timestamps_rel_masked = jnp.where(t_too_large, -1.0, states.t)
            # Select the last state with t <= t_new
            state_idx = jnp.argmax(timestamps_rel_masked, axis=-1)
            state_for_query = jax.tree.map(
                lambda s: s[jnp.arange(state_idx.shape[0]), state_idx], states
            )
            return self.predict(state_for_query, t_new)

        predicted_states = []
        for t_new in timestamps_new_rel:
            predicted_states.append(get_prediction(t_new))

        positions = jnp.stack([ps.x[..., :3] for ps in predicted_states], axis=1)
        velocities = jnp.stack([ps.x[..., 3:6] for ps in predicted_states], axis=1)
        return (
            positions,
            velocities,
        )

    @classmethod
    def _ball_model(cls, x: jax.Array, dt: jax.Array):
        gravity = 9.81
        kd = 0.1062
        km = 0.0057
        pos = x[..., :3]
        vel = x[..., 3:6]
        spin = x[..., 6:9]
        # Velocity zero causes nan's in the norm gradient --> safe norm with eps
        eps = 1e-6
        vel_norm = jnp.sqrt(jnp.sum(vel * vel, axis=-1) + eps)
        return jnp.stack(
            (
                pos[..., 0] + vel[..., 0] * dt,
                pos[..., 1] + vel[..., 1] * dt,
                pos[..., 2] + vel[..., 2] * dt - 0.5 * gravity * dt**2,
                vel[..., 0]
                + dt
                * (
                    -kd * vel_norm * vel[..., 0]
                    + km * (spin[..., 1] * vel[..., 2] - spin[..., 2] * vel[..., 1])
                ),
                vel[..., 1]
                + dt
                * (
                    -kd * vel_norm * vel[..., 1]
                    + km * (spin[..., 2] * vel[..., 0] - spin[..., 0] * vel[..., 2])
                ),
                vel[..., 2]
                + dt
                * (
                    -kd * vel_norm * vel[..., 2]
                    + km * (spin[..., 0] * vel[..., 1] - spin[..., 1] * vel[..., 0])
                    - gravity
                ),
                spin[..., 0],
                spin[..., 1],
                spin[..., 2],
            ),
            axis=-1,
        )

    @classmethod
    def _observation_model(cls, x: jax.Array):
        return x[..., :3]
