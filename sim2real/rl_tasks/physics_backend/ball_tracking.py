import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax.tree_util import register_pytree_node_class

logger = logging.getLogger(__name__)
try:
    import pam_vicon.o80
    import spatial_transformation
    import tennicam_client
    from ball_launcher_beepy import BallLauncherClient
except ModuleNotFoundError:
    logger.warning("Could not import o80 packages.")


from sim2real.util.ball_ekf import BallEKF, EKFState
from sim2real.rl_tasks.spatial import (
    BALL_IN_CUP_Z_OFFSET_STRING_ATTACH,
    Transformation,
    ball_pos_vel_cup_frame,
    vicon_to_mujoco_frame,
)
from sim2real.rl_tasks.physics_backend.physics_backend import ObjData


@register_pytree_node_class
class CupBallTrackingState:
    def __init__(
        self,
        positions: jax.Array,
        timestamps: jax.Array,
    ):
        self.positions = positions
        self.timestamps = timestamps

    @classmethod
    def init(
        cls,
        position: jax.Array,
        timestamp: jax.Array,
        buffer_length: int,
    ) -> "CupBallTrackingState":
        positions = position[..., None, :].repeat(buffer_length, axis=-2)
        timestamps = timestamp[..., None].repeat(buffer_length, axis=-1)
        return cls(positions, timestamps)

    def advance(
        self,
        position: jnp.ndarray,
        timestamp: jnp.ndarray,
    ) -> "CupBallTrackingState":
        # If there is no new position for the current step, write the last position to the buffer again.
        nan_position = jnp.any(jnp.isnan(position), axis=-1)
        new_position = jnp.where(
            nan_position[..., None],
            self.positions[..., -1, :],
            position,
        )
        new_timestamp = jnp.where(
            nan_position,
            self.timestamps[..., -1],
            timestamp,
        )
        positions = self.positions.at[..., :-1, :].set(self.positions[..., 1:, :])
        positions = positions.at[..., -1, :].set(new_position.astype(jnp.float32))
        timestamps = self.timestamps.at[..., :-1].set(self.timestamps[..., 1:])
        timestamps = timestamps.at[..., -1].set(new_timestamp.astype(jnp.float32))
        return CupBallTrackingState(positions, timestamps)

    def get_obj_data(self) -> ObjData:
        curr_pos = self.positions[..., -1, :]
        curr_ts = self.timestamps[..., -1]
        first_pos = self.positions[..., 0, :]
        first_ts = self.timestamps[..., 0]
        ball_vel = jnp.where(
            (curr_ts - first_ts > 1e-6)[..., None],
            (curr_pos - first_pos) / (curr_ts[..., None] - first_ts[..., None]),
            0.0,
        )
        quat = (
            jnp.zeros((*curr_pos.shape[:-1], 4), dtype=jnp.float32).at[..., 1].set(1.0)
        )
        rotvel = jnp.zeros_like(ball_vel)
        return ObjData(curr_pos, quat, ball_vel, rotvel, curr_ts)

    def tree_flatten(self):
        children = (self.positions, self.timestamps)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class CupBallTrackingReal:
    def __init__(
        self,
        buffer_length: int,
    ):
        self.buffer_length = buffer_length

        self.vicon_frontend = pam_vicon.o80.FrontEnd("vicon")
        vicon_obs = self.vicon_frontend.latest()
        vicon_frame = vicon_obs.get_extended_state()
        robot_base = vicon_frame.subjects[pam_vicon.o80.Subjects.ROBOT_GREEN_BASE]
        self.robot_pose = spatial_transformation.Transformation.from_cpp(
            robot_base.global_pose
        )
        self.init_ball_timestamp = vicon_obs.get_time_stamp()

    def get_raw_data(
        self, retry: bool = False, verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        # Do-while loop if retry is True
        obs = self.vicon_frontend.latest()
        while True:
            ball_pos = (
                obs.get_extended_state()
                .subjects[pam_vicon.o80.Subjects.CUP_BALL]
                .global_pose.translation
            )
            if np.all(np.abs(ball_pos) < 1e-8):
                ball_pos = np.full(3, np.nan)
            if not retry or not np.any(np.isnan(ball_pos)):
                break
            else:
                if verbose:
                    logger.warning("No ball detected. Retrying...")
                obs = self.vicon_frontend.latest()
        ball_pos_mujoco_frame, _ = vicon_to_mujoco_frame(
            ball_pos, np.array([0.0, 0.0, 0.0, 1.0]), self.robot_pose
        )
        timestamp = (obs.get_time_stamp() - self.init_ball_timestamp) * 1e-9
        return ball_pos_mujoco_frame, timestamp

    def reset(
        self,
        ball_pos: jax.Array,
        ball_vel: jax.Array,
        ball_timestamp: jax.Array,
    ) -> Tuple[CupBallTrackingState, ObjData]:
        state = CupBallTrackingState.init(ball_pos, ball_timestamp, self.buffer_length)
        return state, state.get_obj_data()

    def update(
        self,
        state: "CupBallTrackingState",
        ball_pos: jax.Array,
        ball_timestamp: jax.Array,
        ee_data: ObjData,
    ) -> Tuple[CupBallTrackingState, ObjData]:
        ball_pos = self._check_cup_ball_data(ball_pos, ball_timestamp, state, ee_data)
        state = state.advance(ball_pos, ball_timestamp)
        return state, state.get_obj_data()

    @classmethod
    def _check_cup_ball_data(
        cls,
        ball_pos: jax.Array,
        ball_timestamp: jax.Array,
        state: CupBallTrackingState,
        ee_data: ObjData,
    ) -> jax.Array:
        # Check that the current ball velocity (difference to the last position) is not too large.
        velocity = (ball_pos - state.positions[..., -1, :]) / (
            (ball_timestamp[..., None] - state.timestamps[..., -1, None])
        )
        velocity_too_large = jnp.linalg.norm(velocity, axis=-1, keepdims=True) > 50.0
        ball_pos_cup_frame, _ = ball_pos_vel_cup_frame(ee_data, ball_pos, None)
        sting_attach_pos = jnp.array(
            [[0.0, 0.0, BALL_IN_CUP_Z_OFFSET_STRING_ATTACH]], dtype=jnp.float32
        )
        dist_to_string_attach_too_large = (
            jnp.linalg.norm(
                ball_pos_cup_frame - sting_attach_pos, axis=-1, keepdims=True
            )
            > 0.3
        )
        position_invalid = jnp.logical_or(
            velocity_too_large, dist_to_string_attach_too_large
        )
        return jnp.where(position_invalid, jnp.full_like(ball_pos, jnp.nan), ball_pos)


class CupBallTrackingSim:
    def __init__(
        self,
        buffer_length: int,
        sim_ball_tracking_noise_std: float,
        sim_ball_tracking_dropout_rate: float,
    ):
        self.buffer_length = buffer_length
        self.sim_ball_tracking_noise_std = sim_ball_tracking_noise_std
        self.sim_ball_tracking_dropout_rate = sim_ball_tracking_dropout_rate

    def reset(
        self,
        ball_pos: jax.Array,
        ball_vel: jax.Array,
        ball_timestamp: jax.Array,
        rng_randomization: Optional[jax.random.PRNGKey],
    ) -> Tuple[CupBallTrackingState, ObjData]:
        ball_pos = self._apply_sim_noise(ball_pos, rng_randomization)
        state = CupBallTrackingState.init(ball_pos, ball_timestamp, self.buffer_length)
        return state, state.get_obj_data()

    def update(
        self,
        state: "CupBallTrackingState",
        ball_pos: jax.Array,
        ball_timestamp: jax.Array,
        rng_randomization: Optional[jax.random.PRNGKey],
    ) -> Tuple[CupBallTrackingState, ObjData]:
        if rng_randomization is not None:
            rng_noise, rng_dropout = jax.random.split(rng_randomization)
            ball_pos = self._apply_sim_noise(ball_pos, rng_noise)
            ball_pos = self._apply_sim_dropout(ball_pos, rng_dropout)
        state = state.advance(ball_pos, ball_timestamp)
        return state, state.get_obj_data()

    def _apply_sim_noise(
        self, ball_pos: jax.Array, rng: jax.random.PRNGKey
    ) -> jax.Array:
        ball_pos_noise = (
            jax.random.normal(rng, ball_pos.shape, dtype=jnp.float32)
            * self.sim_ball_tracking_noise_std
        )
        return ball_pos + ball_pos_noise

    def _apply_sim_dropout(
        self, ball_pos: jax.Array, rng: jax.random.PRNGKey
    ) -> jax.Array:
        ball_tracking_dropout_mask = jax.random.choice(
            rng,
            jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
            shape=(*ball_pos.shape[:-1], 1),
            p=jnp.array(
                [
                    self.sim_ball_tracking_dropout_rate,
                    1.0 - self.sim_ball_tracking_dropout_rate,
                ]
            ),
        )
        return ball_pos * ball_tracking_dropout_mask


class TableTennisBallTrackingReal:
    def __init__(self):
        self.tennicam_frontend = tennicam_client.FrontEnd("tennicam_client")
        self.ball_launcher = BallLauncherClient("10.42.26.171", 5555)

        # TODO: Should also use Vicon for the table tennis task
        self.robot_pose = Transformation(
            np.array([0.24172509, -3.38384842, -0.95054469]),
            scipy.spatial.transform.Rotation.identity(),
        )
        self.ball_ekf = BallEKF()
        self.last_ball_iteration = -1

    def get_raw_data(
        self, retry: bool = False, verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        if self.tennicam_frontend is None:
            raise NotImplementedError(
                "O80Backend: Table tennis task only supported on the real system"
            )
        # Do-while loop if retry is True
        obs = self.tennicam_frontend.latest()
        ball_detected = False
        ball_pos = np.full(3, np.nan)
        while not ball_detected:
            ball_detected = (
                obs.get_ball_id() != -1
                and obs.get_iteration() != self.last_ball_iteration
            )
            if ball_detected:
                ball_pos = np.array(obs.get_position())
                self.last_ball_iteration = obs.get_iteration()
            else:
                if not retry:
                    break
                if verbose:
                    logger.warning("No ball detected. Retrying...")
                obs = self.tennicam_frontend.latest()

        ball_pos_mujoco_frame, _ = vicon_to_mujoco_frame(
            ball_pos, np.array([0.0, 0.0, 0.0, 1.0]), self.robot_pose
        )
        timestamp = obs.get_time_stamp() * 1e-9
        return np.array(ball_pos_mujoco_frame, dtype=np.float32), timestamp

    def reset(
        self,
        ball_pos: jax.Array,
        ball_vel: jax.Array,
        ball_timestamp: jax.Array,
    ) -> Tuple[EKFState, ObjData]:
        state = self.ball_ekf.reset(ball_pos, ball_vel, ball_timestamp)
        return state, self._get_ball_data(state)

    def update(
        self,
        state: EKFState,
        ball_pos: jax.Array,
        ball_timestamp: jax.Array,
    ) -> Tuple[EKFState, ObjData]:
        updated_state = self.ball_ekf.update(state, ball_pos, ball_timestamp)
        state = jax.tree.map(
            lambda so, su: jnp.where(
                jnp.any(jnp.isnan(ball_pos)).reshape((-1,) + (1,) * (so.ndim - 1)),
                so,
                su,
            ),
            state,
            updated_state,
        )
        # TODO: Use the model's predictions for the current timestamp
        return state, self._get_ball_data(state)

    def get_ball_pos(self, state: EKFState, timestamp: jax.Array) -> ObjData:
        state_pred = self.ball_ekf.predict(state, timestamp)
        return self._get_ball_data(state_pred)

    @classmethod
    def _get_ball_data(cls, state: EKFState) -> ObjData:
        pos = state.x[:, :3]
        quat = jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32).repeat(
            pos.shape[0], axis=0
        )
        linvel = state.x[:, 3:6]
        rotvel = jnp.zeros_like(linvel)
        return ObjData(pos, quat, linvel, rotvel, state.t)
