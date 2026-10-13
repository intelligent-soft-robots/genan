import logging
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import o80
import o80_pam
import o80_pam_wrp
import pam_interface
import pam_mujoco
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from mujoco import mjx

from sim2real.actuator_network.models import ActuatorEnsemble
from sim2real.data_collection.cocontraction import (
    Cocontraction,
)
from sim2real.rl_tasks.physics_backend.ball_tracking import (
    CupBallTrackingReal,
    TableTennisBallTrackingReal,
)
from sim2real.rl_tasks.physics_backend.exceptions import WrongTimestampException
from sim2real.rl_tasks.physics_backend.physics_backend import (
    Buffers,
    ObjData,
    PhysicsBackend,
)
from sim2real.util.model import create_model_spec, quat_from_mujoco

logger = logging.getLogger(__name__)


@register_pytree_node_class
class O80BackendState:
    def __init__(
        self,
        timestamp: jax.Array,
        iteration: jax.Array,
        init_iteration: jax.Array,
        iterations_per_step: jax.Array,
        ball_tracking_state: Any,
    ):
        self.timestamp = timestamp
        self.iteration = iteration
        self.init_iteration = init_iteration
        self.iterations_per_step = iterations_per_step
        self.ball_tracking_state = ball_tracking_state

    @classmethod
    def init(
        cls,
        timestamp: jax.Array,
        iteration: jax.Array,
        iterations_per_step: jax.Array,
        ball_tracking_state: Any,
    ) -> "O80BackendState":
        return O80BackendState(
            timestamp, iteration, iteration, iterations_per_step, ball_tracking_state
        )

    def update(
        self, timestamp: jax.Array, iteration: jax.Array, ball_tracking_state: Any
    ) -> "O80BackendState":
        return O80BackendState(
            timestamp,
            iteration,
            self.init_iteration,
            self.iterations_per_step,
            ball_tracking_state,
        )

    def tree_flatten(self):
        children = (
            self.timestamp,
            self.iteration,
            self.init_iteration,
            self.iterations_per_step,
            self.ball_tracking_state,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def step(self) -> jax.Array:
        return (self.iteration - self.init_iteration) // self.iterations_per_step


class O80Backend(PhysicsBackend):
    def __init__(
        self,
        real_robot: bool,
        dt: float,
        mujoco_id: str,
        history_length: int,
        ndp: Sequence[float] = (0.5, 0.3, 0.4, 0.3),
        simulate_encoder_quantization: bool = False,
        table_position: Sequence[float] = (0.6, 1.5, 0.76),
        graphics: bool = False,
        exception_on_incorrect_timestamp: bool = True,
        force_real_time_mode: bool = False,
        context: Literal["racket", "ball_in_cup"] = "racket",
        stop_before_each_episode: bool = False,
        ball_pos_buffer_length: int = 10,
        ball_launcher_phi: float = 0.49,
        ball_launcher_theta: float = 0.75,
        ball_launcher_motor_actuation=0.35,
    ):
        if real_robot:
            assert (
                not simulate_encoder_quantization
            ), "simulate_encoder_quantization enabled for real robot."
            self.robot_frontend = o80_pam.FrontEnd("real_robot")
            assert (
                self.robot_frontend.get_frequency() == 1000.0
            ), f"Expected frontend frequency of 1000Hz, got {self.robot_frontend.get_frequency()}."
            self.mujoco_handle = None
            robot_config = pam_interface.Pamy2DefaultConfiguration(False)
        else:
            assert (
                context == "racket"
            ), f"Context {context} not supported in simulation."
            robot_segment_id = "robot"
            robot = pam_mujoco.MujocoRobot(
                pam_mujoco.RobotType.PAMY2,
                robot_segment_id,
                control=pam_mujoco.MujocoRobot.PRESSURE_CONTROL,
            )
            self.mujoco_handle = pam_mujoco.MujocoHandle(
                mujoco_id,
                robot1=robot,
                graphics=graphics,
                accelerated_time=not force_real_time_mode,
                burst_mode=not force_real_time_mode,
            )
            self.robot_frontend = self.mujoco_handle.frontends[robot_segment_id]
            robot_config = pam_interface.JsonConfiguration(str(robot.json_control_path))
        self.cocontraction = Cocontraction.from_robot_config(robot_config, ndp)

        robot_timestep = 1.0 / self.robot_frontend.get_frequency()
        assert (
            dt % robot_timestep < 1e-6
        ), f"target timestep ({dt}s) is not a multiple of robot timestep ({robot_timestep}s)"
        self.robot_iterations_per_step = round(dt / robot_timestep)

        self.exception_on_incorrect_timestamp = exception_on_incorrect_timestamp

        # Model needed for forward kinematics
        model = create_model_spec(context).compile()
        self.mjx_model = mjx.put_model(model)
        self.mjx_state = mjx.make_data(model)
        self.context = context
        self.stop_before_each_episode = stop_before_each_episode

        if real_robot and context == "ball_in_cup":
            self.ball_tracking = CupBallTrackingReal(ball_pos_buffer_length)
        elif real_robot and context == "table_tennis":
            from ball_launcher_beepy import BallLauncherClient

            self.ball_launcher = BallLauncherClient("10.42.26.171", 5555)
            self.ball_tracking = TableTennisBallTrackingReal()

        self.ball_launcher_phi = ball_launcher_phi
        self.ball_launcher_theta = ball_launcher_theta
        self.ball_launcher_motor_actuation = ball_launcher_motor_actuation

        super().__init__(
            dt,
            history_length,
            simulate_encoder_quantization,
            table_position,
        )

    def _reset(
        self,
        rng: jax.random.PRNGKey,
        control: jnp.ndarray,
        positions: Optional[jnp.ndarray],
        prev_positions: Optional[jnp.ndarray],
        obj_frames: Sequence[Tuple[str, Optional[str]]],
        obj_poses: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
        ensemble_size: Optional[ActuatorEnsemble] = None,
        dry_run: jnp.ndarray = jnp.array(False),
        **kwargs,
    ) -> Tuple[
        ArrayLike, ArrayLike, Dict[str, ObjData], O80BackendState, Dict[str, Any]
    ]:
        init_pressures_ago, init_pressures_antago = (
            self.cocontraction.control_to_pressures(control[0])
        )

        cb_return_shape = (
            jnp.zeros(4, dtype=jnp.float32),
            jnp.zeros(4, dtype=jnp.float32),
            (
                jnp.zeros(3, dtype=jnp.float32)
                if self.context in ["ball_in_cup", "table_tennis"]
                else None
            ),
            (
                jnp.zeros(3, dtype=jnp.float32)
                if self.context in ["ball_in_cup", "table_tennis"]
                else None
            ),
            (
                jnp.array(0.0, dtype=jnp.float32)
                if self.context in ["ball_in_cup", "table_tennis"]
                else None
            ),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )
        (
            position,
            prev_position,
            ball_pos,
            ball_vel,
            ball_timestamp,
            timestamp,
            iteration,
        ) = io_callback(
            self._reset_cb,
            cb_return_shape,
            init_pressures_ago,
            init_pressures_antago,
            dry_run,
            ordered=True,
        )
        if self.context == "ball_in_cup" or self.context == "table_tennis":
            ball_tracking_state, ball_data = self.ball_tracking.reset(
                ball_pos[None], ball_vel[None], jnp.zeros((1,), dtype=jnp.float32)
            )
            velocity = (position - prev_position) / self.dt
            if self.context == "ball_in_cup":
                obj_data = {
                    "ball": ball_data,
                    "cup": self._add_batch_axis(
                        self._get_ee_data(position, velocity, "cup")
                    ),
                }
            else:
                obj_data = {
                    "ball": ball_data,
                    "racket": self._add_batch_axis(
                        self._get_ee_data(position, velocity, "racket")
                    ),
                }
        else:
            obj_data = {}
            ball_tracking_state = None
        robot_iterations_per_step = round(self.dt * self.robot_frontend.get_frequency())
        state = O80BackendState.init(
            timestamp,
            iteration,
            robot_iterations_per_step,
            ball_tracking_state,
        )
        return (
            position[None],
            prev_position[None],
            obj_data,
            state,
            {
                "iteration": iteration[None, None],
                "timestamp": timestamp[None, None],
                "step": state.step[None, None],
            },
        )

    def _reset_cb(
        self,
        init_pressures_ago: jnp.ndarray,
        init_pressures_antago: jnp.ndarray,
        dry_run: jnp.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float],
        float,
        int,
    ]:
        init_pressures_ago = np.array(init_pressures_ago)
        init_pressures_antago = np.array(init_pressures_antago)
        if dry_run:
            observation = self.robot_frontend.latest()
            prev_observation = self.robot_frontend.latest()
            if self.context == "ball_in_cup" or self.context == "table_tennis":
                ball_pos = np.zeros(3, dtype=jnp.float32)
                ball_vel = np.zeros_like(ball_pos)
                ball_timestamp = 0.0
            else:
                ball_pos = None
                ball_vel = None
                ball_timestamp = None
        else:
            if self.mujoco_handle is not None:
                self.mujoco_handle.reset()
                if self.mujoco_handle._burster_client is not None:
                    self.mujoco_handle.burst()

            if self.context == "ball_in_cup":
                self.wait(0.5)
                self.go_to_control(np.array([0.0, -1.0, -0.5, 0.0]), 2.0)
                self.wait(1.0)
                self.go_to_control(np.array([0.0, 1.0, 1.0, 0.0]), 4.0)
                self.wait(2.0)
            elif self.context == "table_tennis":
                self.go_to_control(np.array([0.0, -0.5, -0.2, 0.0]), 2.0)

            if self.stop_before_each_episode:
                if self.context != "ball_in_cup":
                    self.go_to_control(np.array([0.0, 0.0, 0.0, 0.0]), 2.0)
                input("Press enter to continue...")

            self.go_to_pressures(init_pressures_ago, init_pressures_antago, 2.0)
            self.wait(1.0)

            if self.context == "ball_in_cup":
                ball_pos, ball_timestamp = self.ball_tracking.get_raw_data(retry=True)
                # TODO: Return the actual velocity
                ball_vel = np.full_like(ball_pos, np.nan)
            elif self.context == "table_tennis":
                self.ball_launcher.set_state(
                    self.ball_launcher_phi,
                    self.ball_launcher_theta,
                    self.ball_launcher_motor_actuation,
                    self.ball_launcher_motor_actuation,
                    self.ball_launcher_motor_actuation,
                )
                self.wait(2.0)
                self.ball_launcher.launch_ball()
                self.wait(0.25)
                prev_ball_pos, prev_ball_timestamp = self.ball_tracking.get_raw_data(
                    retry=True
                )
                ball_pos, ball_timestamp = self.ball_tracking.get_raw_data(retry=True)
                ball_vel = (ball_pos - prev_ball_pos) / (
                    ball_timestamp - prev_ball_timestamp
                )
            else:
                ball_pos = None
                ball_vel = None
                ball_timestamp = None

            last_observations = self.robot_frontend.get_latest_observations(
                round(self.dt * 1000) + 1
            )
            observation = last_observations[-1]
            prev_observation = last_observations[0]

        position = np.array(observation.get_positions(), dtype=np.float32)
        prev_position = np.array(prev_observation.get_positions(), dtype=np.float32)
        timestamp = self._extract_timestamp(observation)
        iteration = observation.get_iteration()

        return (
            position,
            prev_position,
            ball_pos,
            ball_vel,
            ball_timestamp,
            timestamp,
            iteration,
        )

    def _step(
        self,
        rng: jax.random.PRNGKey,
        buffers: Buffers,
        state: O80BackendState,
        actuator_ensemble: Optional[ActuatorEnsemble] = None,
        dry_run: jnp.ndarray = jnp.array(False),
        **kwargs,
    ) -> Tuple[ArrayLike, Dict[str, ObjData], O80BackendState, Dict[str, Any]]:
        pressure_ago, pressure_antago = self.cocontraction.control_to_pressures(
            buffers.controls[0, -1]
        )

        cb_return_shape = (
            jnp.zeros(4, dtype=jnp.float32),
            (
                jnp.zeros(3, dtype=jnp.float32)
                if self.context in ["ball_in_cup", "table_tennis"]
                else None
            ),
            (
                jnp.array(0.0, dtype=jnp.float32)
                if self.context in ["ball_in_cup", "table_tennis"]
                else None
            ),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )
        position, ball_pos, ball_timestamp, timestamp, iteration = io_callback(
            self._step_cb,
            cb_return_shape,
            pressure_ago,
            pressure_antago,
            state,
            dry_run,
            ordered=True,
        )

        velocity = (position - buffers.positions[0, -1]) / self.dt
        if self.context == "ball_in_cup":
            ee_data = self._get_ee_data(position, velocity, "cup")
            ball_tracking_state, ball_data = self.ball_tracking.update(
                state.ball_tracking_state, ball_pos[None], ball_timestamp[None], ee_data
            )
            obj_data = {
                "ball": ball_data,
                "cup": self._add_batch_axis(ee_data),
            }
        elif self.context == "table_tennis":
            ball_tracking_state, _ = self.ball_tracking.update(
                state.ball_tracking_state,
                ball_pos[None],
                ((state.step + 1) * self.dt)[None],
            )
            ball_data = self.ball_tracking.get_ball_pos(
                state.ball_tracking_state, ((state.step + 1) * self.dt)[None]
            )
            obj_data = {
                "ball": ball_data,
                "racket": self._add_batch_axis(
                    self._get_ee_data(position, velocity, "racket")
                ),
            }
        else:
            ball_tracking_state = None
            obj_data = {}

        return (
            position[None],
            obj_data,
            state.update(timestamp, iteration, ball_tracking_state),
            {
                "iteration": iteration[None, None],
                "timestamp": timestamp[None, None],
                "step": state.step[None, None],
            },
        )

    def _step_cb(
        self,
        pressure_ago: jnp.ndarray,
        pressure_antago: jnp.ndarray,
        state: O80BackendState,
        dry_run: jnp.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], float, int]:
        pressure_ago = np.array(pressure_ago)
        pressure_antago = np.array(pressure_antago)
        timestamp = state.timestamp.item()
        if dry_run:
            observation = self.robot_frontend.latest()
        else:
            self.robot_frontend.add_command(
                pressure_ago, pressure_antago, o80.Mode.OVERWRITE
            )
            self.robot_frontend.pulse()

            while (
                (self._extract_timestamp(self.robot_frontend.latest()) - timestamp)
            ) % 60.0 - self.dt < -1e-4:
                if (
                    self.mujoco_handle is not None
                    and self.mujoco_handle._burster_client is not None
                ):
                    self.mujoco_handle.burst()
            observation = self.robot_frontend.latest()
            if (
                self._extract_timestamp(observation) - timestamp
            ) % 60.0 - self.dt > 1e-4:
                error_msg = (
                    f"Expected timestep {self.dt:.3f}, got "
                    f"{(self._extract_timestamp(observation) - timestamp) % 60.0:.4f}"
                )
                if self.exception_on_incorrect_timestamp:
                    raise WrongTimestampException(error_msg)
                else:
                    logger.warning(error_msg)
        position = np.array(observation.get_positions(), dtype=np.float32)
        timestamp = self._extract_timestamp(observation)
        iteration = observation.get_iteration()

        if self.context == "ball_in_cup":
            ball_pos, ball_timestamp = self.ball_tracking.get_raw_data()
        elif self.context == "table_tennis":
            if dry_run:
                ball_pos = np.zeros(3, dtype=np.float32)
                ball_timestamp = 0.0
            else:
                ball_pos, ball_timestamp = self.ball_tracking.get_raw_data()
        else:
            ball_pos = None
            ball_timestamp = None
        return (
            position,
            ball_pos,
            ball_timestamp,
            timestamp,
            iteration,
        )

    def _extract_timestamp(self, observation: o80_pam_wrp.Observation) -> float:
        timestamp_int = observation.get_extended_state().get_time_stamp()
        # Converting the timestamps to float32 to make sure that there are no precision differences to state.timestamp.
        if self.mujoco_handle is not None:
            return np.float32(timestamp_int * 1e-9)
        else:
            return np.float32(timestamp_int * 1e-6)

    def go_to_control(self, control: np.ndarray, time: float) -> None:
        pressures_ago, pressures_antago = self.cocontraction.control_to_pressures(
            control
        )
        self.go_to_pressures(pressures_ago, pressures_antago, time)

    def go_to_pressures(
        self, pressure_ago: np.ndarray, pressure_antago: np.ndarray, time: float
    ) -> None:
        self.robot_frontend.add_command(
            pressure_ago,
            pressure_antago,
            o80.Duration_us.milliseconds(round(time * 1000.0)),
            o80.Mode.OVERWRITE,
        )
        self.robot_frontend.pulse()
        self.wait(time)

    def wait(self, time: float):
        init_timestep = self._extract_timestamp(self.robot_frontend.latest())
        while (
            self._extract_timestamp(self.robot_frontend.latest()) - init_timestep
        ) % 60.0 < time:
            if (
                self.mujoco_handle is not None
                and self.mujoco_handle._burster_client is not None
            ):
                self.mujoco_handle.burst()

    def _get_ee_data(
        self, joint_position: jnp.ndarray, joint_velocity: jnp.ndarray, ee_name: str
    ) -> ObjData:
        curr_state = self.mjx_state.replace(
            qpos=self.mjx_state.qpos.at[:4].set(joint_position),
            qvel=self.mjx_state.qvel.at[:4].set(joint_velocity),
        )
        curr_state = mjx.forward(self.mjx_model, curr_state)
        ee_id = mujoco.mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        ee_pos = curr_state.xpos[ee_id]
        Jp, Jr = mujoco.mjx.jac(self.mjx_model, curr_state, ee_pos, ee_id)
        linvel = (jnp.swapaxes(Jp, -1, -2) @ curr_state.qvel[:, None]).squeeze(-1)
        rotvel = (jnp.swapaxes(Jr, -1, -2) @ curr_state.qvel[:, None]).squeeze(-1)
        return ObjData(
            ee_pos.astype(jnp.float32),
            quat_from_mujoco(curr_state.xquat[ee_id].astype(jnp.float32)),
            linvel.astype(jnp.float32),
            rotvel.astype(jnp.float32),
            None,
        )

    @classmethod
    def _add_batch_axis(cls, obj_data: ObjData) -> ObjData:
        return ObjData(
            obj_data.pos[None],
            obj_data.quat[None],
            obj_data.linvel[None],
            obj_data.rotvel[None],
            obj_data.timestamp[None] if obj_data.timestamp is not None else None,
        )
