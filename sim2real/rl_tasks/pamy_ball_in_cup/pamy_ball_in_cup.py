import logging
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.spatial.transform
import numpy as np
from jax.scipy.spatial.transform import Rotation
from jax.tree_util import register_pytree_node_class

from sim2real.rl_tasks.pamy_env import PamyEnv, PamyEnvState
from sim2real.rl_tasks.physics_backend.physics_backend import (
    Buffers,
    ObjData,
    PhysicsBackend,
)
from sim2real.rl_tasks.spatial import (
    BALL_IN_CUP_STRING_LENGTH,
    BALL_IN_CUP_Z_OFFSET_STRING_ATTACH,
    ball_pos_vel_cup_frame,
)

logger = logging.getLogger(__name__)


@register_pytree_node_class
class PamyBallInCupState:
    def __init__(
        self,
        last_success_reward: jnp.ndarray,
        last_alignment_reward: jnp.ndarray,
        last_ball_above_cup_reward: jnp.ndarray,
        success: jnp.ndarray,
        episode_metrics: Dict[str, jnp.ndarray],
    ):
        self.success_reward = last_success_reward
        self.alignment_reward = last_alignment_reward
        self.ball_above_cup_reward = last_ball_above_cup_reward
        self.success = success
        self.episode_metrics = episode_metrics

    @classmethod
    def init(
        cls,
        num_instances: int,
        episode_metrics: Dict[str, jnp.ndarray],
    ) -> "PamyBallInCupState":
        zeros = jnp.zeros(num_instances, dtype=jnp.float32)
        zeros_bool = zeros.astype(bool)
        return cls(
            zeros,
            zeros,
            zeros,
            zeros_bool,
            episode_metrics,
        )

    def update(
        self,
        success_reward: jnp.ndarray,
        alignment_reward: jnp.ndarray,
        ball_above_cup_reward: jnp.ndarray,
        ball_in_cup: jnp.ndarray,
        curr_step_metrics: Dict[str, jnp.ndarray],
    ) -> "PamyBallInCupState":
        episode_metrics = {
            k: ep + curr_step_metrics[k] for k, ep in self.episode_metrics.items()
        }
        return PamyBallInCupState(
            success_reward,
            alignment_reward,
            ball_above_cup_reward,
            ball_in_cup,
            episode_metrics,
        )

    def tree_flatten(self):
        children = (
            self.success_reward,
            self.alignment_reward,
            self.ball_above_cup_reward,
            self.success,
            self.episode_metrics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "PamyBallInCupState":
        return cls(*children)


class PamyBallInCupEnv(PamyEnv):
    def __init__(
        self,
        backend: PhysicsBackend,
        num_instances: int = 1024,
        episode_length_s: float = 1.0,
        action_repeat: int = 1,
        num_history_states: int = 3,
        step_size: int = 1,
        delta_history: bool = True,
        max_control_diff: float = 0.01,
        history_observations: bool = True,
        velocity_observations: bool = True,
        reset_ctrl_lower: Sequence[float] = (-0.5, -0.8, -0.5, -0.2),
        reset_ctrl_upper: Sequence[float] = (0.5, -0.6, 0.0, 0.2),
        # The limits of the robot's kinematics (the robot is penalized for reaching these limits).
        joint_limits_lower_deg: Sequence[float] = (-90.0, -90.0, -90.0, -90.0),
        joint_limits_upper_deg: Sequence[float] = (90.0, 90.0, 90.0, 90.0),
        action_penalty_weight: float = 0.0,
        velocity_penalty_weight: float = 0.0,
        an_disagreement_penalty_weight: float = 0.0,
        success_reward_weight: float = 10.0,
        alignment_reward_weight: float = 0.0,
        ball_above_cup_reward_weight: float = 0.0,
        success_max_cup_angle_deg: float = 180.0,
        ball_init_max_polar_angle_deg: float = 180.0,
        terminate_on_success: bool = False,
    ):
        self.reset_ctrl_lower = jnp.array(reset_ctrl_lower, dtype=jnp.float32)
        self.reset_ctrl_upper = jnp.array(reset_ctrl_upper, dtype=jnp.float32)

        super().__init__(
            backend,
            num_instances,
            episode_length_s,
            action_repeat,
            np.full(6, -np.inf, dtype=np.float32),
            np.full(6, np.inf, dtype=np.float32),
            num_history_states,
            step_size,
            delta_history,
            max_control_diff,
            history_observations,
            velocity_observations,
            False,
            True,
            joint_limits_lower_deg,
            joint_limits_upper_deg,
            action_penalty_weight,
            velocity_penalty_weight,
            an_disagreement_penalty_weight,
        )

        self.success_reward_weight = success_reward_weight
        self.alignment_reward_weight = alignment_reward_weight
        self.ball_above_cup_reward_weight = ball_above_cup_reward_weight

        self.success_max_cup_angle = success_max_cup_angle_deg * jnp.pi / 180.0
        self.ball_init_max_polar_angle = ball_init_max_polar_angle_deg * jnp.pi / 180.0
        assert (
            not terminate_on_success or num_instances == 1
        ), f"terminate_on_success only supported for num_instances=1, got {num_instances}."
        self.terminate_on_success = terminate_on_success

    def task_initial(
        self,
        rng: jax.random.PRNGKey,
        dry_run: bool,
    ) -> Tuple[
        PamyBallInCupState, Buffers, Dict[str, ObjData], Any, Dict[str, jnp.ndarray]
    ]:
        rng_pos, rng_ball_polar, rng_ball_azimuth, rng_ball_quat, rng_goal = (
            jax.random.split(rng, 5)
        )

        controls = jax.random.uniform(
            rng_pos,
            (self.num_instances, self.num_dofs),
            minval=self.reset_ctrl_lower,
            maxval=self.reset_ctrl_upper,
            dtype=jnp.float32,
        )
        string_attach_pos = jnp.array(
            [[0.0, 0.0, BALL_IN_CUP_Z_OFFSET_STRING_ATTACH]], dtype=jnp.float32
        )
        # Sample in cosine space to avoid biasing the samples toward the poles 🇵🇱.
        cos_ball_polar_angle_min = jnp.cos(self.ball_init_max_polar_angle)
        cos_ball_polar_angle = jax.random.uniform(
            rng_ball_polar,
            self.num_instances,
            minval=cos_ball_polar_angle_min,
            maxval=1.0,
            dtype=jnp.float32,
        )
        r = BALL_IN_CUP_STRING_LENGTH - 1e-6
        ball_z = r * cos_ball_polar_angle
        r_xy = jnp.sqrt(r**2 - ball_z**2)
        # Sample from [-0.8 pi, 0.8 pi] to avoid positions in which the ball collides with the link.
        ball_azimuth = jax.random.uniform(
            rng_ball_azimuth,
            self.num_instances,
            minval=-0.8 * jnp.pi,
            maxval=0.8 * jnp.pi,
            dtype=jnp.float32,
        )
        ball_pos_rel = jnp.stack(
            [r_xy * jnp.cos(ball_azimuth), r_xy * jnp.sin(ball_azimuth), ball_z],
            axis=-1,
        )
        ball_pos = ball_pos_rel + string_attach_pos
        ball_quat = jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32).repeat(
            self.num_instances, axis=0
        )
        buffers, obj_data, backend_state, backend_info = self.backend.reset(
            rng,
            controls,
            None,
            None,
            obj_frames=(("ball", "cup"),),
            obj_poses=((ball_pos, ball_quat),),
            dry_run=dry_run,
        )

        episode_metrics_init = {
            "rewards/success_reward": jnp.array(0.0, dtype=jnp.float32),
            "rewards/alignment_reward": jnp.array(0.0, dtype=jnp.float32),
            "rewards/ball_above_cup_reward": jnp.array(0.0, dtype=jnp.float32),
        }

        state = PamyBallInCupState.init(
            self.num_instances,
            episode_metrics_init,
        )

        return state, buffers, obj_data, backend_state, backend_info

    def task_transition(
        self,
        actions: jnp.ndarray,
        state: PamyEnvState,
        new_buffers: Buffers,
        new_obj_data: Dict[str, ObjData],
        new_backend_state: Any,
        rng: jax.random.PRNGKey,
        **kwargs,
    ) -> PamyBallInCupState:
        success = self.is_success(new_obj_data)
        success_reward = self.success_reward_weight * success.astype(jnp.float32)
        ball_pos_rel, _ = ball_pos_vel_cup_frame(
            new_obj_data["cup"], new_obj_data["ball"].pos, new_obj_data["ball"].linvel
        )
        # Dot product between offset ball - cup (normalized to length 1) and cup normal vector in cup coordinate system
        # --> Cup normal vector = (0, 0, 1)^T
        ball_cup_alignment = ball_pos_rel[:, 2] / jnp.linalg.norm(ball_pos_rel, axis=1)
        alignment_reward = jnp.where(
            jnp.bitwise_not(success),
            self.alignment_reward_weight * ball_cup_alignment,
            0.0,
        )

        ball_above_cup = jnp.bitwise_and(
            jnp.linalg.norm(ball_pos_rel[..., :2], axis=-1) < 0.055,
            ball_pos_rel[..., 2] > 0.0,
        )
        max_dist_above_cup = (
            BALL_IN_CUP_STRING_LENGTH + BALL_IN_CUP_Z_OFFSET_STRING_ATTACH
        )
        ball_above_cup_reward = jnp.where(
            jnp.bitwise_and(jnp.bitwise_not(success), ball_above_cup),
            self.ball_above_cup_reward_weight
            * (max_dist_above_cup - ball_pos_rel[..., 2])
            / max_dist_above_cup,
            0.0,
        )

        curr_step_metrics = {
            "rewards/success_reward": success_reward.mean(),
            "rewards/alignment_reward": alignment_reward.mean(),
            "rewards/ball_above_cup_reward": ball_above_cup_reward.mean(),
        }

        return state.task.update(
            success_reward,
            alignment_reward,
            ball_above_cup_reward,
            success,
            curr_step_metrics,
        )

    def task_observation(
        self,
        state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        ball_pos_rel, ball_vel_rel = ball_pos_vel_cup_frame(
            state.obj_data["cup"],
            state.obj_data["ball"].pos,
            state.obj_data["ball"].linvel,
        )

        return jnp.concatenate((ball_pos_rel, ball_vel_rel), axis=1)

    def task_reward(
        self,
        state: PamyEnvState,
        action: jnp.ndarray,
        next_state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        return next_state.task.success_reward + next_state.task.alignment_reward

    def task_terminal(self, next_state: PamyEnvState) -> jnp.ndarray:
        return jnp.logical_and(
            self.terminate_on_success, self.is_success(next_state.obj_data)
        )

    def task_episode_metrics(self, state: PamyEnvState) -> Dict[str, Any]:
        episode_metrics = {
            "ball_in_cup/success_rate": jnp.mean(state.task.success),
            # float since ints are not logged to tensorboard by skrl
            "ball_in_cup/success_count": jnp.sum(state.task.success).astype(
                jnp.float32
            ),
        }
        for k, v in state.task.episode_metrics.items():
            episode_metrics[k] = v / state.backend.step
        return episode_metrics

    def is_success(self, obj_data: Dict[str, ObjData]) -> jnp.ndarray:
        ball_pos_rel, _ = ball_pos_vel_cup_frame(
            obj_data["cup"],
            obj_data["ball"].pos,
            obj_data["ball"].linvel,
        )
        horizontal_diff = jnp.linalg.norm(ball_pos_rel[..., :2], axis=-1)
        height_diff = ball_pos_rel[..., 2]
        ball_in_cup = jnp.bitwise_and(
            horizontal_diff < 0.06,
            jnp.bitwise_and(-0.07 < height_diff, height_diff < 0.005),
        )
        cup_normal = Rotation.from_quat(obj_data["cup"].quat).apply(
            jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
        )
        cup_upright = cup_normal[..., 2] > jnp.cos(self.success_max_cup_angle)
        return jnp.logical_and(ball_in_cup, cup_upright)
