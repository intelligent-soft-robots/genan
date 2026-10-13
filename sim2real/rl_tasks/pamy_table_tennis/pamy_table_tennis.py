import logging
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.spatial.transform
import numpy as np
from jax.tree_util import register_pytree_node_class

from sim2real.rl_tasks.pamy_env import PamyEnv, PamyEnvState
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackendState

try:
    from sim2real.rl_tasks.physics_backend.o80_backend import O80Backend
except:
    O80Backend = None
from sim2real.rl_tasks.physics_backend.physics_backend import (
    Buffers,
    ObjData,
    PhysicsBackend,
)
from sim2real.util.table_tennis import (
    BALL_RADIUS,
    ball_pos_vel_rel_racket,
)

logger = logging.getLogger(__name__)


@register_pytree_node_class
class PamyTableTennisState:
    def __init__(
        self,
        ball_hit_time: jax.Array,
        landing_point_table_frame: jax.Array,
        last_ball_hit_reward: jax.Array,
        last_goal_dist_reward: jax.Array,
        episode_metrics: Dict[str, jax.Array],
    ):
        self.ball_hit_time = ball_hit_time
        self.landing_point_table_frame = landing_point_table_frame
        self.ball_hit_reward = last_ball_hit_reward
        self.goal_dist_reward = last_goal_dist_reward
        self.episode_metrics = episode_metrics

    @classmethod
    def init(
        cls,
        num_instances: int,
        episode_metrics: Dict[str, jax.Array],
    ) -> "PamyTableTennisState":
        zeros = jnp.zeros(num_instances, dtype=jnp.float32)
        nans = jnp.full(num_instances, jnp.nan, dtype=jnp.float32)
        nans_3d = jnp.full((num_instances, 3), jnp.nan, dtype=jnp.float32)
        return cls(
            nans,
            nans_3d,
            zeros,
            zeros,
            episode_metrics,
        )

    def update(
        self,
        ball_hit_time: jax.Array,
        landing_point_table_frame: jax.Array,
        ball_hit_reward: jax.Array,
        goal_dist_reward: jax.Array,
        curr_step_metrics: Dict[str, jax.Array],
    ) -> "PamyTableTennisState":
        episode_metrics = {
            k: ep + curr_step_metrics[k] for k, ep in self.episode_metrics.items()
        }
        return PamyTableTennisState(
            jnp.where(jnp.isnan(self.ball_hit_time), ball_hit_time, self.ball_hit_time),
            landing_point_table_frame,
            ball_hit_reward,
            goal_dist_reward,
            episode_metrics,
        )

    def tree_flatten(self):
        children = (
            self.ball_hit_time,
            self.landing_point_table_frame,
            self.ball_hit_reward,
            self.goal_dist_reward,
            self.episode_metrics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "PamyTableTennisState":
        return cls(*children)


class PamyTableTennisEnv(PamyEnv):
    def __init__(
        self,
        backend: PhysicsBackend,
        num_instances: int = 1024,
        episode_length_s: float = 1.2,
        action_repeat: int = 1,
        num_history_states: int = 3,
        step_size: int = 1,
        delta_history: bool = True,
        max_control_diff: float = 0.01,
        history_observations: bool = True,
        velocity_observations: bool = True,
        finite_horizon_time_obs: bool = True,
        reset_ctrl_lower: Sequence[float] = (-0.5, -0.8, -0.5, -0.2),
        reset_ctrl_upper: Sequence[float] = (0.5, -0.6, 0.0, 0.2),
        # The limits of the robot's kinematics (the robot is penalized for reaching these limits).
        joint_limits_lower_deg: Sequence[float] = (-90.0, -90.0, -90.0, -90.0),
        joint_limits_upper_deg: Sequence[float] = (90.0, 90.0, 90.0, 90.0),
        action_penalty_weight: float = 0.0,
        velocity_penalty_weight: float = 0.0,
        an_disagreement_penalty_weight: float = 0.0,
        ball_hit_reward_weight: float = 0.0,
        goal_dist_reward_weight: float = 0.0,
        goal_pos_table_xy: Sequence[float] = (0.0, 0.7),
        ball_pos_obs_world_frame: bool = False,
        end_effector_obs: bool = False,
    ):
        self.reset_ctrl_lower = jnp.array(reset_ctrl_lower, dtype=jnp.float32)
        self.reset_ctrl_upper = jnp.array(reset_ctrl_upper, dtype=jnp.float32)
        obs_lim_low = np.array([-np.inf] * 6 + [0.0], dtype=jnp.float32)
        obs_lim_high = np.array([np.inf] * 6 + [1.0], dtype=jnp.float32)
        if end_effector_obs:
            obs_lim_low = np.concatenate(
                (obs_lim_low, np.full(6, -np.inf, dtype=jnp.float32))
            )
            obs_lim_high = np.concatenate(
                (obs_lim_high, np.full(6, np.inf, dtype=jnp.float32))
            )

        super().__init__(
            backend,
            num_instances,
            episode_length_s,
            action_repeat,
            obs_lim_low,
            obs_lim_high,
            num_history_states,
            step_size,
            delta_history,
            max_control_diff,
            history_observations,
            velocity_observations,
            True,
            finite_horizon_time_obs,
            joint_limits_lower_deg,
            joint_limits_upper_deg,
            action_penalty_weight,
            velocity_penalty_weight,
            an_disagreement_penalty_weight,
        )

        self.ball_hit_reward_weight = ball_hit_reward_weight
        self.goal_dist_reward_weight = goal_dist_reward_weight
        self.goal_pos_table_xy = jnp.array(goal_pos_table_xy, dtype=jnp.float32)
        self.ball_pos_obs_world_frame = ball_pos_obs_world_frame
        self.end_effector_obs = end_effector_obs

    def task_initial(
        self,
        rng: jax.Array,
        dry_run: bool,
    ) -> Tuple[
        PamyTableTennisState, Buffers, Dict[str, ObjData], Any, Dict[str, jax.Array]
    ]:
        rng_ctrl, rng = jax.random.split(rng)
        controls = jax.random.uniform(
            rng_ctrl,
            (self.num_instances, self.num_dofs),
            minval=self.reset_ctrl_lower,
            maxval=self.reset_ctrl_upper,
            dtype=jnp.float32,
        )
        buffers, obj_data, backend_state, backend_info = self.backend.reset(
            rng,
            controls,
            None,
            None,
            obj_frames=(),
            obj_poses=(),
            dry_run=dry_run,
        )

        episode_metrics_init = {
            "rewards/ball_hit_reward": jnp.array(0.0, dtype=jnp.float32),
            "rewards/goal_dist_reward": jnp.array(0.0, dtype=jnp.float32),
        }

        state = PamyTableTennisState.init(
            self.num_instances,
            episode_metrics_init,
        )

        return state, buffers, obj_data, backend_state, backend_info

    def task_transition(
        self,
        actions: jax.Array,
        state: PamyEnvState,
        new_buffers: Buffers,
        new_obj_data: Dict[str, ObjData],
        new_backend_state: Any,
        rng: jax.Array,
        dry_run: bool = False,
        **kwargs,
    ) -> PamyTableTennisState:
        if isinstance(new_backend_state, MjxBackendState):
            ball_hit = jnp.logical_and(
                new_backend_state.tt_ball_post_racket_contact,
                jnp.isnan(state.task.ball_hit_time),
            )
        else:
            ball_hit = jnp.zeros(self.num_instances, dtype=bool)
        post_ball_hit = ~jnp.isnan(state.task.ball_hit_time)
        # Ball hit reward is only given once
        ball_hit_reward = self.ball_hit_reward_weight * ball_hit.astype(jnp.float32)

        last_step = state.backend.step + 1 >= self.max_episode_length
        if O80Backend is None or not isinstance(self.backend, O80Backend):
            landing_point_table_frame = jax.lax.cond(
                last_step,
                self.get_landing_point,
                lambda _1, _2, _3: jnp.full(
                    (self.num_instances, 3), jnp.nan, dtype=jnp.float32
                ),
                new_backend_state,
                new_buffers,
                dry_run,
            )
            goal_offset_xy = landing_point_table_frame[:, :2] - self.goal_pos_table_xy

            goal_dist_reward = jnp.where(
                jnp.logical_and(
                    ~jnp.any(jnp.isnan(goal_offset_xy), axis=-1),
                    jnp.logical_or(ball_hit, post_ball_hit),
                ),
                self.calc_goal_reward(goal_offset_xy),
                0.0,
            )
        else:
            landing_point_table_frame = jnp.full(
                (self.num_instances, 3), jnp.nan, dtype=jnp.float32
            )
            goal_dist_reward = jnp.full(self.num_instances, 0.0, dtype=jnp.float32)

        curr_step_metrics = {
            "rewards/ball_hit_reward": ball_hit_reward.mean(),
            "rewards/goal_dist_reward": goal_dist_reward.mean(),
        }

        ball_hit_time = jnp.where(ball_hit, new_backend_state.timestamp, jnp.nan)
        return state.task.update(
            ball_hit_time,
            landing_point_table_frame,
            ball_hit_reward,
            goal_dist_reward,
            curr_step_metrics,
        )

    def calc_goal_reward(self, goal_offset: jax.Array) -> jax.Array:
        goal_dist = jnp.linalg.norm(goal_offset, axis=1)
        # TODO: Where is that constant coming from?
        goal_dist_rew = 1.0 - ((goal_dist / 2.95942) ** 0.75)
        rtt_cap = 0.0  # TODO: Why is this -1.2 in the original implementation?
        return self.goal_dist_reward_weight * jnp.clip(goal_dist_rew, min=rtt_cap)

    def get_landing_point(
        self, backend_state: PamyEnvState, buffers: Buffers, dry_run: bool
    ) -> jax.Array:
        control = buffers.controls[:, -1]

        def update_landing_point(carry: Tuple[jax.Array, Any]) -> Tuple[jax.Array, Any]:
            _landing_point, _backend_state = carry
            _, obj_data, _backend_state, _ = self.backend.step(
                jax.random.PRNGKey(42),
                control,
                buffers,
                _backend_state,
                enable_obj_tracking_randomization=False,
                dry_run=dry_run,
            )
            ball_position_table_frame = self._to_table_frame(obj_data["ball"].pos)
            landing_point_empty = jnp.all(jnp.isnan(_landing_point), axis=1)
            tolerance = 0.001
            _landing_point = jnp.where(
                jnp.logical_and(
                    landing_point_empty,
                    ball_position_table_frame[:, 2] < BALL_RADIUS + tolerance,
                )[:, None],
                ball_position_table_frame,
                _landing_point,
            )
            return _landing_point, _backend_state

        landing_point, _ = jax.lax.while_loop(
            lambda carry: jnp.any(jnp.all(jnp.isnan(carry[0]), axis=1)),
            update_landing_point,
            (
                jnp.full((self.num_instances, 3), jnp.nan, dtype=jnp.float32),
                backend_state,
            ),
        )
        return landing_point

    def _to_table_frame(self, position: jax.Array) -> jax.Array:
        return position - self.backend.table_position

    def task_observation(
        self,
        state: PamyEnvState,
        rng: jax.Array,
        params: None = None,
    ) -> jax.Array:
        if self.ball_pos_obs_world_frame:
            ball_pos = state.obj_data["ball"].pos
            ball_vel = state.obj_data["ball"].linvel
        else:
            ball_pos, ball_vel = ball_pos_vel_rel_racket(state.obj_data, None)
        post_ball_hit = ~jnp.isnan(state.task.ball_hit_time)
        obs = jnp.concatenate(
            (
                ball_pos,
                ball_vel,
                post_ball_hit.astype(jnp.float32)[:, None],
            ),
            axis=1,
        )
        if self.end_effector_obs:
            obs = jnp.concatenate(
                (obs, state.obj_data["racket"].pos, state.obj_data["racket"].linvel),
                axis=-1,
            )
        return obs

    def task_reward(
        self,
        state: PamyEnvState,
        action: jax.Array,
        next_state: PamyEnvState,
        rng: jax.Array,
        params: None = None,
    ) -> jax.Array:
        return next_state.task.ball_hit_reward + next_state.task.goal_dist_reward

    def task_episode_metrics(self, state: PamyEnvState) -> Dict[str, Any]:
        goal_dist = jnp.linalg.norm(
            state.task.landing_point_table_frame[:, :2] - self.goal_pos_table_xy, axis=1
        )
        post_ball_hit = ~jnp.isnan(state.task.ball_hit_time)
        episode_metrics = {
            "table_tennis/ball_hit_rate": jnp.mean(post_ball_hit),
            # float since ints are not logged to tensorboard by skrl
            "table_tennis/ball_hit_count": jnp.sum(post_ball_hit).astype(jnp.float32),
            "table_tennis/ball_hit_time": jnp.mean(
                state.task.ball_hit_time, where=~jnp.isnan(state.task.ball_hit_time)
            ),
            "table_tennis/avg_goal_dist": jnp.mean(goal_dist.mean(0)),
        }
        for succ_dist in [0.685, 0.4, 0.2, 0.1, 0.05]:
            episode_metrics[f"table_tennis/success_rate_dist_{succ_dist}"] = jnp.mean(
                goal_dist <= succ_dist
            )
        for k, v in state.task.episode_metrics.items():
            episode_metrics[k] = v / state.backend.step
        return episode_metrics
