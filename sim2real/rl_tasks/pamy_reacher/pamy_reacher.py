import logging
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sim2real.rl_tasks.pamy_env import PamyEnv, PamyEnvState
from sim2real.rl_tasks.physics_backend.physics_backend import (
    Buffers,
    ObjData,
    PhysicsBackend,
)

logger = logging.getLogger(__name__)


@register_pytree_node_class
class PamyReacherState:
    def __init__(
        self,
        goal_pos: jnp.ndarray,
        last_target_dist_loss: jnp.ndarray,
        ep_steps_goal_dist: jnp.ndarray,
        episode_metrics: Dict[str, jnp.ndarray],
    ):
        self.goal_pos = goal_pos
        self.target_dist_loss = last_target_dist_loss
        self.ep_steps_goal_dist = ep_steps_goal_dist
        self.episode_metrics = episode_metrics

    @classmethod
    def init(
        cls,
        goal_pos: jnp.ndarray,
        max_episode_length: int,
        episode_metrics: Dict[str, jnp.ndarray],
    ) -> "PamyReacherState":
        num_dofs = 4
        num_instances = goal_pos.shape[0]
        zeros = jnp.zeros(num_instances, dtype=jnp.float32)
        ep_steps_goal_dist = jnp.nan * jnp.ones(
            (num_instances, max_episode_length - 1, num_dofs),
            dtype=jnp.float32,
        )
        return cls(goal_pos, zeros, ep_steps_goal_dist, episode_metrics)

    def update(
        self,
        target_dist_loss: jnp.ndarray,
        goal_dist: jnp.ndarray,
        step: int,
        curr_step_metrics: Dict[str, jnp.ndarray],
    ) -> "PamyReacherState":
        episode_metrics = {
            k: ep + curr_step_metrics[k] for k, ep in self.episode_metrics.items()
        }
        return PamyReacherState(
            self.goal_pos,
            target_dist_loss,
            self.ep_steps_goal_dist.at[
                jnp.arange(self.ep_steps_goal_dist.shape[0]), step - 1
            ].set(goal_dist),
            episode_metrics,
        )

    def tree_flatten(self):
        children = (
            self.goal_pos,
            self.target_dist_loss,
            self.ep_steps_goal_dist,
            self.episode_metrics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "PamyReacherState":
        return cls(*children)


class PamyReacherEnv(PamyEnv):
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
        reset_ctrl_lower: Sequence[float] = (-0.7, -0.8, -0.5, -0.8),
        reset_ctrl_upper: Sequence[float] = (0.7, -0.4, 0.5, 0.8),
        # The range to sample the targets from.
        workspace_limits_lower_deg: Sequence[float] = (-70.0, -70.0, -45.0, -20.0),
        workspace_limits_upper_deg: Sequence[float] = (70.0, 60.0, 25.0, 20.0),
        # The limits of the robot's kinematics (the robot is penalized for reaching these limits).
        joint_limits_lower_deg: Sequence[float] = (-90.0, -90.0, -90.0, -90.0),
        joint_limits_upper_deg: Sequence[float] = (90.0, 90.0, 90.0, 90.0),
        action_penalty_weight: float = 0.0,
        velocity_penalty_weight: float = 0.0,
        an_disagreement_penalty_weight: float = 0.0,
        target_dist_loss_norm: int = 1,
        dofs_goal: Sequence[float] = (0, 1, 2, 3),
    ):
        self.dofs_goal = np.array(list(set(dofs_goal)), dtype=np.uint8)

        self.reset_ctrl_lower = np.array(reset_ctrl_lower, dtype=np.float64)
        self.reset_ctrl_upper = np.array(reset_ctrl_upper, dtype=np.float64)
        self.workspace_limits_lower = (
            jnp.array(workspace_limits_lower_deg, dtype=jnp.float32) * jnp.pi / 180.0
        )
        self.workspace_limits_upper = (
            jnp.array(workspace_limits_upper_deg, dtype=jnp.float32) * jnp.pi / 180.0
        )
        lim_goal_pos_low = np.array(
            self.workspace_limits_lower[self.dofs_goal], dtype=np.float32
        )
        lim_goal_pos_high = np.array(
            self.workspace_limits_upper[self.dofs_goal], dtype=np.float32
        )

        super().__init__(
            backend,
            num_instances,
            episode_length_s,
            action_repeat,
            lim_goal_pos_low,
            lim_goal_pos_high,
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

        # Make sure that the workspace limits are not too close to the hard joint limits
        assert jnp.all(
            self.joint_limits_lower
            < self.workspace_limits_lower - self.joint_limits_penalty_dist
        )
        assert jnp.all(
            self.joint_limits_upper
            > self.workspace_limits_upper + self.joint_limits_penalty_dist
        )

        self.target_dist_loss_norm = target_dist_loss_norm

    def task_initial(
        self,
        rng: jax.random.PRNGKey,
        dry_run: bool,
    ) -> Tuple[
        PamyReacherState, Buffers, Dict[str, ObjData], Any, Dict[str, jnp.ndarray]
    ]:
        rng_ctrl, rng_goal = jax.random.split(rng)

        control = jax.random.uniform(
            rng_ctrl,
            (self.num_instances, self.num_dofs),
            minval=self.reset_ctrl_lower,
            maxval=self.reset_ctrl_upper,
            dtype=jnp.float32,
        )
        goal_pos = jax.random.uniform(
            rng_goal,
            (self.num_instances, self.num_dofs),
            minval=self.workspace_limits_lower,
            maxval=self.workspace_limits_upper,
            dtype=jnp.float32,
        )
        buffers, obj_data, backend_state, backend_info = self.backend.reset(
            rng,
            control,
            None,
            None,
            dry_run=dry_run,
        )

        episode_metrics_init = {
            "rewards/target_dist_loss": jnp.array(0, dtype=jnp.float32),
        }

        state = PamyReacherState.init(
            goal_pos,
            self.max_episode_length,
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
    ) -> PamyReacherState:
        pos_normalization = (
            self.workspace_limits_upper[self.dofs_goal]
            - self.workspace_limits_lower[self.dofs_goal]
        )
        position = new_buffers.positions[:, -1]
        target_dist_loss = jnp.linalg.norm(
            (state.task.goal_pos[:, self.dofs_goal] - position[:, self.dofs_goal])
            / pos_normalization,
            ord=self.target_dist_loss_norm,
            axis=1,
        )
        goal_dist = jnp.abs(state.task.goal_pos - position)
        curr_step_metrics = {
            "rewards/target_dist_loss": target_dist_loss.mean(),
        }

        return state.task.update(
            target_dist_loss,
            goal_dist,
            state.backend.step,
            curr_step_metrics,
        )

    def task_observation(
        self,
        state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        return state.task.goal_pos[:, self.dofs_goal]

    def task_reward(
        self,
        state: PamyEnvState,
        action: jnp.ndarray,
        next_state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        return -next_state.task.target_dist_loss

    def task_episode_metrics(self, state: PamyEnvState) -> Dict[str, Any]:
        target_dist_metrics = self._calculate_target_dist_metrics(state)
        episode_metrics = {
            f"target_dist/{k}": v for k, v in target_dist_metrics.items()
        }
        for k, v in state.task.episode_metrics.items():
            episode_metrics[k] = v / state.backend.step
        return episode_metrics

    def _calculate_target_dist_metrics(
        self, next_state: PamyEnvState
    ) -> Dict[str, jnp.ndarray]:
        metrics = {}
        for steps in [1, 10, 100]:
            norm_dists = jnp.linalg.norm(
                next_state.task.ep_steps_goal_dist[:, -steps:, self.dofs_goal], axis=2
            )
            max_dists = jnp.max(
                next_state.task.ep_steps_goal_dist[:, -steps:, self.dofs_goal], axis=2
            )
            for success_dist in [0.05, 0.1, 0.2]:
                successes_norm = jnp.sum(
                    jnp.all(norm_dists <= success_dist, axis=1), axis=0
                ).astype(jnp.int32)
                successes_max = jnp.sum(
                    jnp.all(max_dists <= success_dist, axis=1), axis=0
                ).astype(jnp.int32)
                metrics[f"success_rate_norm_{steps}steps_dist{success_dist}"] = (
                    successes_norm / self.num_instances
                )
                metrics[f"success_rate_max_{steps}steps_dist{success_dist}"] = (
                    successes_max / self.num_instances
                )
            metrics[f"max_dist_{steps}steps/norm"] = norm_dists.max(axis=1).mean()
            metrics[f"max_dist_{steps}steps/max"] = max_dists.max(axis=1).mean()
            mean_dists_dofs = jnp.mean(
                next_state.task.ep_steps_goal_dist[:, -steps:], axis=(0, 1)
            )
            metrics[f"mean_dists_{steps}steps_avg"] = mean_dists_dofs[
                self.dofs_goal
            ].mean()
            for dof in self.dofs_goal:
                metrics[f"mean_dists_{steps}steps_dof{dof}"] = mean_dists_dofs[dof]
        return metrics
