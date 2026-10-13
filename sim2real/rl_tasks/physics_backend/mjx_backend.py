import copy
import functools
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import h5py
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import omegaconf
import scipy.spatial.transform
from jax.scipy.spatial.transform import Rotation
from jax.tree_util import register_pytree_node_class
from mujoco import mjx

from sim2real.actuator_network.models import (
    ActuatorEnsemble,
    restore_actuator_net_checkpoint,
)
from sim2real.actuator_network.pack_an_input import pack_an_input
from sim2real.rl_tasks.physics_backend.ball_tracking import CupBallTrackingSim
from sim2real.rl_tasks.physics_backend.physics_backend import (
    Buffers,
    ObjData,
    PhysicsBackend,
    get_an_timestep,
)
from sim2real.rl_tasks.spatial import Transformation, vicon_to_mujoco_frame
from sim2real.util.ball_ekf import BallEKF
from sim2real.util.config_utils import (
    get_an_input_size_from_cfg,
    get_an_training_config,
)
from sim2real.util.model import create_model_spec, quat_from_mujoco
from sim2real.util.table_tennis import (
    ball_pos_vel_rel_racket,
    ball_racket_contact,
    ball_racket_contact_model,
)

logger = logging.getLogger(__name__)


@register_pytree_node_class
class MjxBackendState:
    def __init__(
        self,
        mjx_data: mujoco.mjx.Data,
        dt: float,
        init_step: jax.Array,
        ball_tracking_state: Any,
        tt_ball_traj_idx: Optional[jax.Array],
        tt_ball_augment_trans: Optional[jax.Array],
        tt_ball_post_racket_contact: Optional[jax.Array],
    ):
        self.mjx_data = mjx_data
        self.dt = dt
        self.init_step = init_step
        self.ball_tracking_state = ball_tracking_state
        self.tt_ball_traj_idx = tt_ball_traj_idx
        self.tt_ball_augment_trans = tt_ball_augment_trans
        self.tt_ball_post_racket_contact = tt_ball_post_racket_contact

    @classmethod
    def init(
        cls,
        context: Literal["ball_in_cup", "cup", "racket", "table_tennis"],
        mjx_data: mujoco.mjx.Data,
        dt: float,
        ball_tracking_state: Any,
        tt_ball_traj_idx: Optional[jax.Array],
        tt_ball_augment_trans: Optional[jax.Array],
    ) -> "MjxBackendState":
        if context == "table_tennis":
            ball_post_racket_contact = jnp.zeros(mjx_data.qpos.shape[0], dtype=bool)
        else:
            ball_post_racket_contact = None
            tt_ball_traj_idx = None
            tt_ball_augment_trans = None
        return cls(
            mjx_data,
            dt,
            cls.get_step(mjx_data.time[0], dt),
            ball_tracking_state,
            tt_ball_traj_idx,
            tt_ball_augment_trans,
            ball_post_racket_contact,
        )

    def update(
        self,
        mjx_data: mujoco.mjx.Data,
        ball_tracking_state: Any,
        tt_ball_racket_contact: Optional[jax.Array],
    ) -> "MjxBackendState":
        if self.tt_ball_post_racket_contact is None:
            ball_post_racket_contact = None
        else:
            ball_post_racket_contact = jnp.logical_or(
                self.tt_ball_post_racket_contact, tt_ball_racket_contact
            )
        return MjxBackendState(
            mjx_data,
            self.dt,
            self.init_step,
            ball_tracking_state,
            self.tt_ball_traj_idx,
            self.tt_ball_augment_trans,
            ball_post_racket_contact,
        )

    def tree_flatten(self):
        children = (
            self.mjx_data,
            self.dt,
            self.init_step,
            self.ball_tracking_state,
            self.tt_ball_traj_idx,
            self.tt_ball_augment_trans,
            self.tt_ball_post_racket_contact,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def get_step(cls, time: jax.Array, dt: float) -> jax.Array:
        return jnp.round(time / dt).astype(int)

    @property
    def step(self) -> jax.Array:
        return self.get_step(self.mjx_data.time[0], self.dt) - self.init_step

    @property
    def timestamp(self) -> jax.Array:
        return self.step * self.dt


def step_mjx(model: mujoco.mjx.Model, mjx_data: mujoco.mjx.Data) -> mujoco.mjx.Data:
    batch_shape = mjx_data.qpos.shape[:-1]
    flatten_dims = len(batch_shape)
    mjx_data = jax.tree.map(
        lambda b: b.reshape((np.prod(batch_shape), *b.shape[flatten_dims:])),
        mjx_data,
    )
    mjx_data = jax.vmap(mjx.step, in_axes=(None, 0))(model, mjx_data)
    return jax.tree.map(lambda b: b.reshape((*batch_shape, *b.shape[1:])), mjx_data)


def forward_mjx(model: mujoco.mjx.Model, mjx_data: mujoco.mjx.Data) -> mujoco.mjx.Data:
    batch_shape = mjx_data.qpos.shape[:-1]
    flatten_dims = len(batch_shape)
    mjx_data = jax.tree.map(
        lambda b: b.reshape((np.prod(batch_shape), *b.shape[flatten_dims:])),
        mjx_data,
    )
    mjx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(model, mjx_data)
    return jax.tree.map(lambda b: b.reshape((*batch_shape, *b.shape[1:])), mjx_data)


def jac_mjx(
    model: mujoco.mjx.Model, mjx_data: mujoco.mjx.Data, pos: jax.Array, obj_id
) -> Tuple[jax.Array, jax.Array]:
    batch_shape = mjx_data.qpos.shape[:-1]
    flatten_dims = len(batch_shape)
    pos = pos.reshape((np.prod(batch_shape), *pos.shape[flatten_dims:]))
    mjx_data = jax.tree.map(
        lambda b: b.reshape((np.prod(batch_shape), *b.shape[flatten_dims:])),
        mjx_data,
    )
    Jp, Jr = jax.vmap(mujoco.mjx.jac, in_axes=(None, 0, 0, None))(
        model, mjx_data, pos, obj_id
    )
    Jp = Jp.reshape((*batch_shape, -1, 3))
    Jr = Jr.reshape((*batch_shape, -1, 3))
    return Jp, Jr


class MjxBackend(PhysicsBackend):
    def __init__(
        self,
        context: Literal["ball_in_cup", "cup", "racket", "table_tennis"],
        actuator_net_exp_name: Optional[str] = None,
        actuator_net_name: Optional[str] = None,
        # Only needed if no actuator network path is passed above
        actuator_net_cfg: Optional[omegaconf.DictConfig] = None,
        actuator_net_timestep: Optional[float] = None,
        fixed_model_idx: Optional[int] = None,
        step_all_models: bool = False,
        stop_gradients_an_input: bool = True,
        simulation_precision: Literal["single", "double"] = "double",
        num_reset_steps: int = 200,
        simulate_encoder_quantization: bool = False,
        simulated_ball_tracking_noise_std: float = 0.0,
        simulated_ball_tracking_dropout_rate: float = 0.0,
        ball_pos_buffer_length: int = 10,
        table_position: Sequence[float] = (0.6, 1.5, 0.76),
        tt_ball_augment_trans_min: Sequence[float] = (0.0, 0.0, 0.0),
        tt_ball_augment_trans_max: Sequence[float] = (0.0, 0.0, 0.0),
        **kwargs,
    ):
        if simulation_precision == "double":
            jax.config.update("jax_enable_x64", True)
            jax.config.update(
                "jax_default_matmul_precision",
                "high",
            )
        self.sim_dtype = (
            jnp.float64 if simulation_precision == "double" else jnp.float32
        )
        int_dtype = jnp.int64 if simulation_precision == "double" else jnp.int32
        self.num_reset_steps = num_reset_steps

        if actuator_net_exp_name is not None and actuator_net_name is not None:
            self.actuator_ensemble = restore_actuator_net_checkpoint(
                actuator_net_exp_name, actuator_net_name, "model"
            )
            self.actuator_ensemble.eval()
            self.actuator_net_cfg = get_an_training_config(
                actuator_net_exp_name, actuator_net_name
            ).actuator_net
        else:
            self.actuator_ensemble = None
            self.actuator_net_cfg = actuator_net_cfg

        history_length = (
            1
            + self.actuator_net_cfg.num_history_states * self.actuator_net_cfg.step_size
        )

        dt = (
            get_an_timestep(actuator_net_exp_name, actuator_net_name)
            if actuator_net_exp_name is not None and actuator_net_name is not None
            else actuator_net_timestep
        )

        super().__init__(
            dt,
            history_length,
            simulate_encoder_quantization,
            table_position,
        )

        self.context = context
        model = create_model_spec(context, table_position=table_position).compile()

        if context == "ball_in_cup":
            self.obj_ids = {
                obj_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                for obj_name in ["ball", "cup"]
            }
            self.ball_tracking = CupBallTrackingSim(
                ball_pos_buffer_length,
                simulated_ball_tracking_noise_std,
                simulated_ball_tracking_dropout_rate,
            )
            self.tt_ball_positions = jnp.zeros((0, 0, 3), dtype=self.sim_dtype)
        elif context == "cup":
            self.obj_ids = {
                "cup": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
            }
            self.tt_ball_positions = jnp.zeros((0, 0, 3), dtype=self.sim_dtype)
            self.ball_tracking = None
        elif context == "racket":
            self.obj_ids = {
                "racket": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket")
            }
            self.tt_ball_positions = jnp.zeros((0, 0, 3), dtype=self.sim_dtype)
            self.ball_tracking = None
        elif context == "table_tennis":
            self.obj_ids = {
                "ball": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"),
                "racket": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "racket"),
            }
            ball_file = h5py.File(
                Path(__file__).parents[3] / "assets" / "ball_trajectories.hdf5", "r"
            )
            dataset_name = "aimy_049_075_035_04"
            max_len_ball_traj = np.max(
                [b["trajectory"].shape[0] for b in ball_file[dataset_name].values()]
            )
            ball_positions = [
                np.array(b["trajectory"]) for b in ball_file[dataset_name].values()
            ]
            ball_positions = np.stack(
                [
                    np.pad(
                        bp,
                        ((0, max_len_ball_traj - bp.shape[0]), (0, 0)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    for bp in ball_positions
                ],
                axis=0,
            )
            # The position of the robot when the ball dataset was recorded
            robot_pose_dataset = Transformation(
                np.array([0.24172509, -3.38384842, -0.95054469]),
                scipy.spatial.transform.Rotation.identity(),
            )
            ball_positions = [
                vicon_to_mujoco_frame(
                    b,
                    np.array([[0.0, 0.0, 0.0, 1.0]]).repeat(b.shape[0], axis=0),
                    robot_pose_dataset,
                )[0]
                for b in ball_positions
                if b.shape[0]
            ]

            timestamps = [
                np.array(b["time_stamps"]) * 1e-6
                for b in ball_file[dataset_name].values()
            ]
            timestamps = np.stack(
                [
                    np.pad(
                        ts,
                        (0, max_len_ball_traj - ts.shape[0]),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    for ts in timestamps
                ],
                axis=0,
            )
            # TODO: Set the env timelimit according to max_timestamp
            ball_ekf = BallEKF()
            timestamps_new = jnp.arange(0.0, np.nanmax(timestamps), self.dt)
            ball_positions_filtered, ball_velocities_filtered = (
                ball_ekf.filter_trajectory(
                    jnp.array(ball_positions), jnp.array(timestamps), timestamps_new
                )
            )
            self.tt_ball_positions = ball_positions_filtered.astype(self.sim_dtype)
            self.tt_ball_velocities = ball_velocities_filtered.astype(self.sim_dtype)
            self.ball_tracking = None
        else:
            raise ValueError(f"Invalid context {context}.")

        self.simulated_ball_tracking_noise_std = simulated_ball_tracking_noise_std
        self.simulated_ball_tracking_dropout_rate = simulated_ball_tracking_dropout_rate
        self.tt_ball_augment_trans_min = jnp.array(tt_ball_augment_trans_min)
        self.tt_ball_augment_trans_max = jnp.array(tt_ball_augment_trans_max)

        self.model = mjx.put_model(model)
        self.reset_data_mjx = mjx.make_data(model)
        # TODO: This should be fixed in MJX
        self.reset_data_mjx = self.reset_data_mjx.replace(
            _impl=self.reset_data_mjx._impl.replace(
                ten_wrapadr=self.reset_data_mjx._impl.ten_wrapadr.astype(int_dtype),
                ten_wrapnum=self.reset_data_mjx._impl.ten_wrapnum.astype(int_dtype),
                wrap_obj=self.reset_data_mjx._impl.wrap_obj.astype(int_dtype),
            )
        )
        self.fixed_model_idx = fixed_model_idx
        self.step_all_models = step_all_models
        self.stop_gradients_an_input = stop_gradients_an_input

    def _reset(
        self,
        rng: jax.random.PRNGKey,
        control: jax.Array,
        position: Optional[jax.Array],
        prev_position: Optional[jax.Array],
        obj_frames: Sequence[Tuple[str, Optional[str]]],
        obj_poses: Sequence[Tuple[jax.Array, jax.Array]],
        ensemble_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[
        jax.Array,
        jax.Array,
        Dict[str, ObjData],
        MjxBackendState,
        Dict[str, Any],
    ]:
        assert ensemble_size is not None or self.actuator_ensemble is not None
        if ensemble_size is None:
            ensemble_size = self.actuator_ensemble.ensemble_size
        input_size = get_an_input_size_from_cfg(self.actuator_net_cfg)

        instances_shape = control.shape[:-1]
        mjx_data = jax.tree.map(
            lambda d: jnp.broadcast_to(
                d.reshape((1,) * len(instances_shape) + d.shape),
                instances_shape + d.shape,
            ),
            self.reset_data_mjx,
        )
        if position is not None and prev_position is not None:
            velocities = (position - prev_position) / self.dt
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[..., : position.shape[-1]]
                .set(position)
                .astype(self.sim_dtype),
                qvel=mjx_data.qvel.at[..., : velocities.shape[-1]]
                .set(velocities)
                .astype(self.sim_dtype),
            )
        else:
            rng, rng_reset_init_pos, rng_reset_steps = jax.random.split(rng, 3)
            reset_init_pos = jax.random.uniform(
                rng_reset_init_pos,
                (*instances_shape, self.num_dofs),
                self.sim_dtype,
                jnp.array([-jnp.pi / 4.0, 0.0, 0.0, -jnp.pi / 4.0]),
                jnp.array([jnp.pi / 4.0, jnp.pi / 2.0, jnp.pi / 2.0, jnp.pi / 4.0]),
            )
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[..., : self.num_dofs].set(reset_init_pos),
                qvel=mjx_data.qvel.at[..., : self.num_dofs].set(0.0),
            )
            mjx_data = forward_mjx(self.model, mjx_data)
            reset_init_buffers = Buffers.init(
                reset_init_pos,
                reset_init_pos,
                control,
                self.history_length,
            )
            dummy_ball_tracking_state = (
                self.ball_tracking.reset(
                    jnp.zeros((*instances_shape, 3)),
                    jnp.zeros((*instances_shape, 3)),
                    jnp.zeros(instances_shape),
                    jax.random.PRNGKey(0),
                )[0]
                if self.ball_tracking is not None
                else None
            )
            # Do a couple of steps to reach a configuration where the combination of position and control is plausible.
            positions_reset, _, _ = self._steps(
                rng_reset_steps,
                control[:, None].repeat(self.num_reset_steps, axis=1),
                reset_init_buffers,
                self._get_obj_poses_velocities(mjx_data),
                MjxBackendState.init(
                    self.context,
                    mjx_data,
                    self.dt,
                    dummy_ball_tracking_state,
                    jnp.zeros(instances_shape, dtype=int),
                    jnp.zeros((*instances_shape, 3)),
                ),
                actuator_ensemble=None,
                dry_run=False,
            )
            position = positions_reset[..., -1, :]
            prev_position = positions_reset[..., -2, :]
            velocities = (position - prev_position) / self.dt
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[..., : position.shape[-1]]
                .set(position)
                .astype(self.sim_dtype),
                qvel=mjx_data.qvel.at[..., : velocities.shape[-1]]
                .set(velocities)
                .astype(self.sim_dtype),
            )
        mjx_data = forward_mjx(self.model, mjx_data)
        for (obj_name, frame_name), (obj_pos, obj_quat) in zip(obj_frames, obj_poses):
            if frame_name is not None:
                frame_id = self.obj_ids[frame_name]
                frame_rot = Rotation.from_quat(
                    quat_from_mujoco(mjx_data.xquat[:, frame_id])
                )
                obj_pos = mjx_data.xpos[:, frame_id] + frame_rot.apply(obj_pos)
                obj_quat = (frame_rot * Rotation.from_quat(obj_quat)).as_quat(
                    scalar_first=True
                )
            obj_pose = jnp.concatenate((obj_pos, obj_quat), axis=-1)
            obj_qpos_adr = self.qpos_adr(obj_name)
            obj_qvel_adr = self.qvel_adr(obj_name)
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[..., obj_qpos_adr : obj_qpos_adr + 7].set(
                    obj_pose
                ),
                qvel=mjx_data.qvel.at[..., obj_qvel_adr : obj_qvel_adr + 6].set(0.0),
            )
        mjx_data = forward_mjx(self.model, mjx_data)

        # Info always needs to have the same shape --> Return an info dict with dummy values
        dummy_info = {
            "episode": {
                "actuator_net_disagreement/std": jnp.zeros(
                    instances_shape, dtype=jnp.float32
                ),
                "actuator_net_disagreement/mean_abs_dev": jnp.zeros(
                    instances_shape, dtype=jnp.float32
                ),
            },
            "actuator_network": {
                "model_indices": 255 * jnp.ones(instances_shape, dtype=jnp.uint8),
                "input": jnp.nan
                * jnp.ones(
                    (*instances_shape, input_size),
                    dtype=jnp.float32,
                ),
                "output": jnp.nan
                * jnp.ones(
                    (ensemble_size, *instances_shape, 4),
                    dtype=jnp.float32,
                ),
                "output_not_destd": jnp.nan
                * jnp.ones(
                    (ensemble_size, *instances_shape, 4),
                    dtype=jnp.float32,
                ),
            },
        }

        # TODO: Need to read the initial ball position from the dataset!
        rng, rng_ball_traj_idx, rng_ball_augment_trans = jax.random.split(rng, 3)
        # Only used if the context is table_tennis
        tt_ball_traj_idx = jax.random.randint(
            rng_ball_traj_idx, instances_shape, 0, self.tt_ball_positions.shape[0] + 1
        )
        tt_ball_augment_trans = jax.random.uniform(
            rng_ball_augment_trans,
            (*instances_shape, 3),
            minval=self.tt_ball_augment_trans_min,
            maxval=self.tt_ball_augment_trans_max,
        )
        if self.context == "table_tennis":
            ball_qpos_adr = self.qpos_adr("ball")
            ball_qvel_adr = self.qvel_adr("ball")
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[:, ball_qpos_adr : ball_qpos_adr + 3].set(
                    self.tt_ball_positions[tt_ball_traj_idx, 0] + tt_ball_augment_trans
                ),
                qvel=mjx_data.qvel.at[:, ball_qvel_adr : ball_qvel_adr + 3].set(
                    self.tt_ball_velocities[tt_ball_traj_idx, 0]
                ),
            )
            mjx_data = forward_mjx(self.model, mjx_data)

        # No object position randomization in the first step to ensure that we always have at least one valid position.
        obj_data = self._get_obj_poses_velocities(mjx_data)
        if self.ball_tracking is not None:
            rng, rng_ball_tracking = jax.random.split(rng)
            ball_tracking_state, ball_data = self.ball_tracking.reset(
                obj_data["ball"].pos,
                obj_data["ball"].linvel,
                obj_data["ball"].timestamp,
                rng_ball_tracking,
            )
            obj_data["ball"] = ball_data
        else:
            ball_tracking_state = None

        state = MjxBackendState.init(
            self.context,
            mjx_data,
            self.dt,
            ball_tracking_state,
            tt_ball_traj_idx,
            tt_ball_augment_trans,
        )

        return (
            position[..., : self.num_dofs],
            prev_position,
            obj_data,
            state,
            dummy_info,
        )

    def _step(
        self,
        rng: jax.random.PRNGKey,
        buffers: Buffers,
        state: MjxBackendState,
        actuator_ensemble: Optional[ActuatorEnsemble] = None,
        enable_obj_tracking_randomization: bool = True,
        **kwargs,
    ) -> Tuple[jax.Array, Dict[str, ObjData], Any, Dict[str, Any]]:
        prev_obj_data = self._get_obj_poses_velocities(state.mjx_data)

        rng, compute_torques_rng, obj_tracking_rng = jax.random.split(rng, 3)
        torque, info_an = self._compute_torques(
            compute_torques_rng,
            actuator_ensemble,
            buffers,
        )
        mjx_data = self._step_torque_no_jit(state.mjx_data, torque)
        if self.context == "table_tennis":
            # For the ball simulation there are three cases:
            # - Pre contact: Use the ball trajectory from the dataset
            # - The contact step: Use the ball_racket_contact_model
            # - After contact: Keep the position that was simulated by MJX

            # TODO: Reading the trajectory ball position from mjx_data is very awkward (requires calling forward twice)
            ball_time_idx = jnp.round(mjx_data.time / self.dt).astype(int)
            ball_qpos_adr = self.qpos_adr("ball")
            ball_qvel_adr = self.qvel_adr("ball")
            simulate_ball = jnp.logical_or(
                state.tt_ball_post_racket_contact,
                ball_time_idx >= self.tt_ball_positions.shape[1],
            )[:, None]
            ball_position = jnp.where(
                simulate_ball,
                mjx_data.qpos[:, ball_qpos_adr : ball_qpos_adr + 3],
                self.tt_ball_positions[state.tt_ball_traj_idx, ball_time_idx]
                + state.tt_ball_augment_trans,
            )
            ball_velocity = jnp.where(
                simulate_ball,
                mjx_data.qvel[:, ball_qvel_adr : ball_qvel_adr + 3],
                self.tt_ball_velocities[state.tt_ball_traj_idx, ball_time_idx],
            )
            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[:, ball_qpos_adr : ball_qpos_adr + 3].set(
                    ball_position
                ),
                qvel=mjx_data.qvel.at[:, ball_qvel_adr : ball_qvel_adr + 3].set(
                    ball_velocity
                ),
            )
            mjx_data = forward_mjx(self.model, mjx_data)

            # Read the updated ball position to check if the racket hit the ball
            # --> If yes, use the simulated ball positions from now on
            new_obj_data = self._get_obj_poses_velocities(mjx_data)
            ball_pos_rel, _ = ball_pos_vel_rel_racket(new_obj_data)
            ball_hit = ball_racket_contact(ball_pos_rel, tolerance=0.0)

            # Make sure to apply the contact model just once
            use_contact_model = jnp.logical_and(
                ball_hit, ~state.tt_ball_post_racket_contact
            )[:, None]
            ball_pos_contact_model, ball_vel_contact_model = ball_racket_contact_model(
                prev_obj_data
            )
            ball_position = jnp.where(
                use_contact_model, ball_pos_contact_model, ball_position
            )
            ball_velocity = jnp.where(
                use_contact_model, ball_vel_contact_model, ball_velocity
            )

            mjx_data = mjx_data.replace(
                qpos=mjx_data.qpos.at[:, ball_qpos_adr : ball_qpos_adr + 3].set(
                    ball_position
                ),
                qvel=mjx_data.qvel.at[:, ball_qvel_adr : ball_qvel_adr + 3].set(
                    ball_velocity
                ),
            )
            mjx_data = forward_mjx(self.model, mjx_data)
        else:
            ball_hit = None

        torques_ensemble = info_an["output"]
        uncertainty_std = torques_ensemble.std(axis=0).mean(axis=1)
        uncertainty_mean_abs_dev = jnp.abs(
            torques_ensemble - torques_ensemble.mean(axis=0, keepdims=True)
        ).mean(axis=(0, 2))

        obj_data = self._get_obj_poses_velocities(mjx_data)
        if self.ball_tracking is not None:
            ball_tracking_state, ball_data = self.ball_tracking.update(
                state.ball_tracking_state,
                obj_data["ball"].pos,
                obj_data["ball"].timestamp,
                obj_tracking_rng if enable_obj_tracking_randomization else None,
            )
            obj_data["ball"] = ball_data
        else:
            ball_tracking_state = None

        info = {"actuator_network": info_an}
        if not self.step_all_models:
            info["episode"] = {
                "actuator_net_disagreement/std": uncertainty_std,
                "actuator_net_disagreement/mean_abs_dev": uncertainty_mean_abs_dev,
            }

        state = state.update(mjx_data, ball_tracking_state, ball_hit)

        return (
            mjx_data.qpos[..., :4].astype(jnp.float32),
            obj_data,
            state,
            info,
        )

    @functools.partial(jax.jit, static_argnames="self")
    def step_torque(
        self,
        mjx_data: mujoco.mjx.Data,
        torque: jax.Array,
    ) -> mujoco.mjx.Data:
        return self._step_torque_no_jit(mjx_data, torque)

    def _step_torque_no_jit(
        self,
        mjx_data: mujoco.mjx.Data,
        torque: jax.Array,
    ) -> mujoco.mjx.Data:
        mjx_data = mjx_data.replace(ctrl=torque.astype(self.sim_dtype))
        mjx_data = step_mjx(self.model, mjx_data)
        return forward_mjx(self.model, mjx_data)

    def _compute_torques(
        self,
        rng: jax.random.PRNGKey,
        actuator_ensemble: ActuatorEnsemble,
        buffers: Buffers,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        if actuator_ensemble is None:
            # TODO: Find a way to avoid copying here
            actuator_ensemble = copy.deepcopy(self.actuator_ensemble)

        rng_inference, rng_model_idx = jax.random.split(rng)
        an_hist_len = (
            self.actuator_net_cfg.step_size * self.actuator_net_cfg.num_history_states
            + 1
        )
        an_input = pack_an_input(
            buffers.positions[..., -(an_hist_len + 1) :, :],
            None,
            buffers.controls[..., -an_hist_len:, :],
            self.actuator_net_cfg,
            self.dt,
        )
        if self.stop_gradients_an_input:
            an_input = jax.lax.stop_gradient(an_input)
        num_instances = buffers.positions.shape[-3]
        if self.step_all_models:
            torques_ensemble, torques_ensemble_not_destd, _ = actuator_ensemble(
                an_input, rng_inference
            )
            torque = torques_ensemble
            info = {}
        else:
            torques_ensemble, torques_ensemble_not_destd, _ = (
                actuator_ensemble.forward_all_no_jit(an_input, rng_inference)
            )
            if self.fixed_model_idx is not None:
                model_indices = (
                    jnp.ones((num_instances,), dtype=jnp.uint8) * self.fixed_model_idx
                )
            else:
                model_indices = jax.random.randint(
                    rng_model_idx,
                    (num_instances,),
                    0,
                    actuator_ensemble.ensemble_size,
                    dtype=jnp.uint8,
                )
            torque = torques_ensemble[model_indices, jnp.arange(num_instances)]
            info = {"model_indices": model_indices}
        info["input"] = an_input
        info["output"] = torques_ensemble
        info["output_not_destd"] = torques_ensemble_not_destd
        return torque, info

    def _get_obj_poses_velocities(
        self, mjx_data: mujoco.mjx.Data
    ) -> Dict[str, ObjData]:
        obj_data = {}
        for obj_name, obj_id in self.obj_ids.items():
            pos = mjx_data.xpos[..., obj_id, :]
            quat = quat_from_mujoco(mjx_data.xquat[..., obj_id, :])
            Jp, Jr = jac_mjx(self.model, mjx_data, pos, obj_id)
            linvel = (jnp.swapaxes(Jp, -1, -2) @ mjx_data.qvel[..., None]).squeeze(-1)
            rotvel = (jnp.swapaxes(Jr, -1, -2) @ mjx_data.qvel[..., None]).squeeze(-1)
            obj_data[obj_name] = ObjData(
                pos.astype(jnp.float32),
                quat.astype(jnp.float32),
                linvel.astype(jnp.float32),
                rotvel.astype(jnp.float32),
                mjx_data.time.astype(jnp.float32),
            )
        return obj_data

    def qpos_adr(self, body_name: str):
        return self.model.jnt_qposadr[self.model.body_jntadr[self.obj_ids[body_name]]]

    def qvel_adr(self, body_name: str):
        return self.model.body_dofadr[self.obj_ids[body_name]]
