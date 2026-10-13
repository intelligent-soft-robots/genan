import abc
import functools
import logging
from typing import Any, Dict, Sequence, Tuple

import gymnasium.experimental.functional
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.tree_util import register_pytree_node_class

from sim2real.data_processing.finite_differences import backward_differences_first
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend
from sim2real.rl_tasks.physics_backend.physics_backend import (
    Buffers,
    ObjData,
    PhysicsBackend,
)

logger = logging.getLogger(__name__)
sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])


@register_pytree_node_class
class PamyEnvState:
    def __init__(
        self,
        task_state: Any,
        buffers: Buffers,
        obj_data: Dict[str, ObjData],
        backend_state: Any,
        last_control: jnp.ndarray,
        last_joint_limit_loss: jnp.ndarray,
        last_action_penalty: jnp.ndarray,
        last_velocity_penalty: jnp.ndarray,
        last_an_disagreement_penalty: jnp.ndarray,
        backend_info: Dict[str, Any],
        episode_metrics: Dict[str, jnp.ndarray],
    ):
        self.task = task_state
        self.buffers = buffers
        self.obj_data = obj_data
        self.backend = backend_state
        self.last_control = last_control
        self.joint_limit_loss = last_joint_limit_loss
        self.action_penalty = last_action_penalty
        self.an_disagreement_penalty = last_an_disagreement_penalty
        self.velocity_penalty = last_velocity_penalty
        self.backend_info = backend_info
        self.episode_metrics = episode_metrics

    @classmethod
    def init(
        cls,
        task_state: Any,
        buffers: Buffers,
        obj_data: Dict[str, ObjData],
        backend_state: Any,
        backend_info: Dict[str, Any],
        episode_metrics: Dict[str, jnp.ndarray],
    ) -> "PamyEnvState":
        num_instances = buffers.positions.shape[0]
        zeros = jnp.zeros(num_instances, dtype=jnp.float32)
        return cls(
            task_state,
            buffers,
            obj_data,
            backend_state,
            buffers.controls[:, -1],
            zeros,
            zeros,
            zeros,
            zeros,
            backend_info,
            episode_metrics,
        )

    def update(
        self,
        task_state: Any,
        buffers: Buffers,
        obj_data: Dict[str, ObjData],
        backend_state: Any,
        backend_info: Dict[str, Any],
        joint_lim_loss: jnp.ndarray,
        action_penalty: jnp.ndarray,
        velocity_penalty: jnp.ndarray,
        an_disagreement_penalty: jnp.ndarray,
        curr_step_metrics: Dict[str, jnp.ndarray],
    ) -> "PamyEnvState":
        episode_metrics = {
            k: ep + curr_step_metrics[k] for k, ep in self.episode_metrics.items()
        }
        return PamyEnvState(
            task_state,
            buffers,
            obj_data,
            backend_state,
            buffers.controls[:, -1],
            joint_lim_loss,
            action_penalty,
            velocity_penalty,
            an_disagreement_penalty,
            backend_info,
            episode_metrics,
        )

    def tree_flatten(self):
        children = (
            self.task,
            self.buffers,
            self.obj_data,
            self.backend,
            self.last_control,
            self.joint_limit_loss,
            self.action_penalty,
            self.an_disagreement_penalty,
            self.velocity_penalty,
            self.backend_info,
            self.episode_metrics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class PamyEnv(
    gymnasium.experimental.functional.FuncEnv[
        PamyEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, None, Any
    ],
    abc.ABC,
):

    def __init__(
        self,
        backend: PhysicsBackend,
        num_instances: int,
        episode_length_s: float,
        action_repeat: int,
        task_obs_lim_low: np.ndarray,
        task_obs_lim_high: np.ndarray,
        num_history_states: int = 3,
        step_size: int = 1,
        delta_history: bool = True,
        max_control_diff: float = 0.01,
        history_observations: bool = True,
        velocity_observations: bool = True,
        finite_horizon: bool = False,
        finite_horizon_time_obs: bool = True,
        joint_limits_lower_deg: Sequence[float] = (-90.0, -90.0, -90.0, -90.0),
        joint_limits_upper_deg: Sequence[float] = (90.0, 90.0, 90.0, 90.0),
        action_penalty_weight: float = 0.0,
        velocity_penalty_weight: float = 0.0,
        an_disagreement_penalty_weight: float = 0.0,
    ):
        assert backend.history_length == step_size * num_history_states + 1
        assert backend.history_length == step_size * num_history_states + 1
        assert backend.history_length == step_size * num_history_states + 1

        super().__init__()
        self.num_dofs = 4
        self.backend = backend
        self.num_instances = num_instances if isinstance(backend, MjxBackend) else 1
        self.max_episode_length = round(episode_length_s / self.backend.dt)
        assert self.max_episode_length % action_repeat == 0
        self.action_repeat = action_repeat
        self.num_history_states = num_history_states
        self.step_size = step_size
        self.delta_history = delta_history
        self.max_control_diff = max_control_diff
        self.history_observations = history_observations
        self.velocity_observations = velocity_observations
        self.finite_horizon = finite_horizon
        self.finite_horizon_time_obs = finite_horizon_time_obs
        # The distance from the kinematic joint limits at which we start applying the joint limit penalty
        self.joint_limits_penalty_dist = 10.0 * jnp.pi / 180.0
        self.joint_limits_lower = (
            jnp.array(joint_limits_lower_deg, dtype=jnp.float32) * jnp.pi / 180.0
        )
        self.joint_limits_upper = (
            jnp.array(joint_limits_upper_deg, dtype=jnp.float32) * jnp.pi / 180.0
        )
        self.action_penalty_weight = action_penalty_weight
        self.velocity_penalty_weight = velocity_penalty_weight
        self.an_disagreement_penalty_weight = an_disagreement_penalty_weight

        act_lim_low = -np.ones(self.num_dofs, dtype=np.float32)
        act_lim_high = np.ones(self.num_dofs, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(act_lim_low, act_lim_high)
        self.action_spaces = gymnasium.spaces.Box(
            act_lim_low[None].repeat(self.num_instances, axis=0),
            act_lim_high[None].repeat(self.num_instances, axis=0),
        )

        num_obs_history_states = num_history_states if history_observations else 0
        hist_len_pos_vel = num_obs_history_states
        # If the step size is 1, we need one less control history element because the most recent control is already the
        # first element of the history.
        hist_len_ctrl = (
            num_obs_history_states - 1
            if step_size == 1 and num_obs_history_states > 0
            else num_obs_history_states
        )

        lim_state = [np.pi, np.inf, 1] if self.velocity_observations else [np.pi, 1]
        lim_state_high = np.array([lim_state], dtype=np.float32).repeat(
            self.num_dofs, axis=0
        )
        lim_pos_hist_high = np.pi * np.ones(
            (1, hist_len_pos_vel), dtype=np.float32
        ).repeat(self.num_dofs, axis=0)
        lim_vel_hist_high = (
            np.inf
            * np.ones((1, hist_len_pos_vel), dtype=np.float32).repeat(
                self.num_dofs, axis=0
            )
            if self.velocity_observations
            else np.empty((4, 0), dtype=np.float32)
        )
        lim_ctrl_hist_high = np.ones((1, hist_len_ctrl), dtype=np.float32).repeat(
            self.num_dofs, axis=0
        )
        robot_obs_lim_high = np.concatenate(
            (
                lim_state_high,
                lim_pos_hist_high,
                lim_vel_hist_high,
                lim_ctrl_hist_high,
            ),
            axis=1,
        ).reshape(-1)
        lim_time_high = (
            jnp.ones(1, dtype=jnp.float32)
            if finite_horizon and finite_horizon_time_obs
            else jnp.array([])
        )
        robot_obs_lim_high = jnp.concatenate((robot_obs_lim_high, lim_time_high))
        obs_lim_high = np.concatenate((robot_obs_lim_high, task_obs_lim_high))
        lim_state_low = -lim_state_high
        lim_pos_hist_low = -lim_pos_hist_high
        lim_vel_hist_low = -lim_vel_hist_high
        lim_ctrl_hist_low = -lim_ctrl_hist_high
        robot_obs_lim_low = np.concatenate(
            (
                lim_state_low,
                lim_pos_hist_low,
                lim_vel_hist_low,
                lim_ctrl_hist_low,
            ),
            axis=1,
        ).reshape(-1)
        lim_time_low = (
            jnp.zeros(1, dtype=jnp.float32)
            if finite_horizon and finite_horizon_time_obs
            else jnp.array([])
        )
        robot_obs_lim_low = jnp.concatenate((robot_obs_lim_low, lim_time_low))
        obs_lim_low = np.concatenate((robot_obs_lim_low, task_obs_lim_low))
        # TODO: Fix the observation bounds
        self.observation_space = gymnasium.spaces.Box(obs_lim_low, obs_lim_high)
        self.observation_spaces = gymnasium.spaces.Box(
            obs_lim_low[None].repeat(self.num_instances, axis=0),
            obs_lim_high[None].repeat(self.num_instances, axis=0),
        )

        self.action_penalty_weight = action_penalty_weight
        self.velocity_penalty_weight = velocity_penalty_weight
        if an_disagreement_penalty_weight and not isinstance(backend, MjxBackend):
            logger.warning(
                f"The an_disagreement_penalty is only supported for MjxBackend, "
                f"found backend type {type(backend)}. Disabling the penalty."
            )
            self.an_disagreement_penalty_weight = 0.0
        else:
            self.an_disagreement_penalty_weight = an_disagreement_penalty_weight

        self.num_agents = 1  # TODO: Needed?

    def initial(
        self,
        rng: jax.random.PRNGKey,
        dry_run: bool,
    ) -> PamyEnvState:
        episode_metrics_init = {
            "rewards/joint_limit_loss": jnp.array(0, dtype=jnp.float32),
            "rewards/action_penalty": jnp.array(0, dtype=jnp.float32),
            "rewards/velocity_penalty": jnp.array(0, dtype=jnp.float32),
            "rewards/an_disagreement_penalty": jnp.array(0, dtype=jnp.float32),
        }
        task_state, buffers, obj_data, backend_state, backend_info = self.task_initial(
            rng, dry_run
        )

        state = PamyEnvState.init(
            task_state,
            buffers,
            obj_data,
            backend_state,
            backend_info,
            episode_metrics_init,
        )
        return state

    def transition(
        self,
        state: PamyEnvState,
        actions: jnp.ndarray,
        rng: jax.random.PRNGKey,
        params: None = None,
        dry_run: bool = False,
    ) -> PamyEnvState:
        control = jnp.clip(
            state.last_control + actions * self.max_control_diff,
            -1.0,
            1.0,
        )

        buffers, obj_data, backend_state, backend_info = self.backend.step(
            rng,
            control,
            state.buffers,
            state.backend,
            dry_run=dry_run,
        )

        task_state = self.task_transition(
            actions, state, buffers, obj_data, backend_state, rng, dry_run=dry_run
        )

        # Compute the reward terms already here since they are logged in info and reward() cannot modify the state.
        position = buffers.positions[:, -1]

        dist_joint_limits = jnp.minimum(
            self.joint_limits_upper - position,
            position - self.joint_limits_lower,
        )
        max_joint_limit_loss = 1.0
        # At distance joint_limits_penalty_dist from the joint limits, the penalty is 0;
        # at the joint limits, the penalty is max_joint_limit_loss; interpolate linearly.
        joint_lim_loss = jnp.sum(
            jnp.clip(
                -max_joint_limit_loss
                / self.joint_limits_penalty_dist
                * dist_joint_limits
                + max_joint_limit_loss,
                min=0.0,
                max=max_joint_limit_loss,
            ),
            axis=1,
        )

        action_penalty = self.action_penalty_weight * (
            (control - state.last_control) ** 2
        ).mean(axis=1)

        velocity = backward_differences_first(
            buffers.positions[:, -2:], self.backend.dt, keep_len=False
        )[:, 0]
        velocity_penalty = self.velocity_penalty_weight * (velocity**2).mean(axis=1)

        if self.an_disagreement_penalty_weight > 0.0:
            an_disagreement_penalty = (
                self.an_disagreement_penalty_weight
                * backend_info["actuator_network"]["output_not_destd"]
                .std(axis=0)
                .mean(axis=1)
            )
        else:
            an_disagreement_penalty = jnp.zeros(self.num_instances, dtype=jnp.float32)

        curr_step_metrics = {
            "rewards/joint_limit_loss": joint_lim_loss.mean(),
            "rewards/action_penalty": action_penalty.mean(),
            "rewards/velocity_penalty": velocity_penalty.mean(),
            "rewards/an_disagreement_penalty": an_disagreement_penalty.mean(),
        }

        state = state.update(
            task_state,
            buffers,
            obj_data,
            backend_state,
            backend_info,
            joint_lim_loss,
            action_penalty,
            velocity_penalty,
            an_disagreement_penalty,
            curr_step_metrics,
        )
        return state

    @functools.partial(
        nnx.jit,
        static_argnames=("self",),
        in_shardings=sharding,
        out_shardings=sharding,
    )
    def reset(
        self, rng: jax.random.PRNGKey, dry_run: bool = False
    ) -> Tuple[Tuple[jnp.ndarray, Dict[str, Any]], PamyEnvState]:
        state = self.initial(rng, dry_run=dry_run)
        obs = self.observation(state, rng)
        info = self.state_info(state)
        if "backend" in info:
            backend_info = info["backend"]
            if "episode" in backend_info:
                for k, v in backend_info["episode"].items():
                    info["episode"][f"backend/{k}"] = v.mean()
                info["backend"] = {
                    k: v for k, v in backend_info.items() if k != "episode"
                }
        return (obs, info), state

    @functools.partial(
        nnx.jit,
        static_argnames=("self",),
        in_shardings=sharding,
        out_shardings=sharding,
    )
    def step(
        self,
        action: jnp.ndarray,
        state: PamyEnvState,
        rng: jax.random.PRNGKey,
        dry_run: bool = False,
    ) -> Tuple[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]],
        PamyEnvState,
    ]:
        # To make PyLint happy
        next_state = None
        observation = None
        info = None

        reward = jnp.zeros(self.num_instances, dtype=jnp.float32)
        terminated = jnp.zeros(self.num_instances, dtype=bool)
        truncated = jnp.zeros(self.num_instances, dtype=bool)
        for _ in range(self.action_repeat):
            next_state = self.transition(state, action, rng, dry_run=dry_run)
            observation = self.observation(next_state, rng)
            reward += self.reward(state, action, next_state, rng)
            terminated = jnp.logical_or(terminated, self.terminal(next_state, rng))
            truncated = jnp.logical_or(truncated, self.truncated(next_state))
            info = self.transition_info(state, action, next_state)
            if "backend" in info:
                backend_info = info["backend"]
                if "episode" in backend_info:
                    for k, v in backend_info["episode"].items():
                        info["episode"][f"backend/{k}"] = v.mean()
                    info["backend"] = {
                        k: v for k, v in backend_info.items() if k != "episode"
                    }
            state = next_state
        return (observation, reward, terminated, truncated, info), next_state

    def _maybe_add_time_observation(
        self, state: PamyEnvState, obs: jax.Array
    ) -> jax.Array:
        if self.finite_horizon and self.finite_horizon_time_obs:
            time_obs = jnp.array(
                [[state.backend.step / self.max_episode_length]]
            ).repeat(self.num_instances, axis=0)
            return jnp.concatenate((obs, time_obs), axis=1)
        else:
            return obs

    def observation(
        self,
        state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        num_dofs = 4
        buffers = self.backend.process_buffers_for_obs(state.buffers)
        velocities = backward_differences_first(
            buffers.positions, self.backend.dt, keep_len=False
        )
        velocity_obs = (
            velocities[:, -1:]
            if self.velocity_observations
            else jnp.empty((self.num_instances, 0, num_dofs), dtype=jnp.float32)
        )
        robot_observation = [
            buffers.positions[:, -1:],
            velocity_obs,
            buffers.controls[:, -1:],
        ]
        if self.history_observations:
            position_context = buffers.positions[:, 1 : -1 : self.step_size]
            velocity_context = (
                velocities[:, : -1 : self.step_size]
                if self.velocity_observations
                else jnp.empty((self.num_instances, 0, num_dofs), dtype=jnp.float32)
            )
            # Rightmost control is c_(t-1), while rightmost position and velocity are p_t / v_t
            control_context = buffers.controls[:, 1 : -1 : self.step_size]
            if self.delta_history:
                position_context -= buffers.positions[:, -1:]
                if self.velocity_observations:
                    velocity_context -= velocities[:, -1:]
                control_context -= buffers.controls[:, -1:]

            robot_observation.extend(
                [
                    position_context,
                    velocity_context,
                    control_context,
                ]
            )
        robot_observation = (
            jnp.concatenate(robot_observation, axis=1)
            .swapaxes(1, 2)
            .reshape(self.num_instances, -1)
        )
        robot_observation = self._maybe_add_time_observation(state, robot_observation)
        task_observation = self.task_observation(state, rng, params)
        return jnp.concatenate((robot_observation, task_observation), axis=1)

    def reward(
        self,
        state: PamyEnvState,
        action: jnp.ndarray,
        next_state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        task_reward = self.task_reward(state, action, next_state, rng)
        # The robot reward terms are computed in transition()
        return (
            task_reward
            - next_state.joint_limit_loss
            - next_state.action_penalty
            - next_state.velocity_penalty
            - next_state.an_disagreement_penalty
        )

    def truncated(self, next_state: PamyEnvState) -> jnp.ndarray:
        return jnp.array(
            [
                not self.finite_horizon
                and next_state.backend.step >= self.max_episode_length
            ]
        ).repeat(self.num_instances)

    def terminal(
        self, next_state: PamyEnvState, rng: Any, params: None = None
    ) -> jnp.ndarray:
        terminated_timelimit = jnp.array(
            [self.finite_horizon and next_state.backend.step >= self.max_episode_length]
        ).repeat(self.num_instances)
        terminated_task = self.task_terminal(next_state)
        return jnp.logical_or(terminated_timelimit, terminated_task)

    def transition_info(
        self,
        state: PamyEnvState,
        action: jnp.ndarray,
        next_state: PamyEnvState,
        params: None = None,
    ) -> Dict[str, Any]:
        return self.state_info(next_state)

    def state_info(self, state: PamyEnvState, params: None = None) -> Dict[str, Any]:
        episode_metrics = self.task_episode_metrics(state)

        for k, v in state.episode_metrics.items():
            episode_metrics[k] = v / state.backend.step

        return {
            "episode": episode_metrics,
            "backend": state.backend_info,
            "position": state.buffers.positions[:, -1],
            "velocity": backward_differences_first(
                state.buffers.positions[:, -2:], self.backend.dt, keep_len=False
            )[:, 0],
            "control": state.buffers.controls[:, -1],
        }

    def close(self):
        self.backend.close()

    @abc.abstractmethod
    def task_initial(
        self,
        rng: jax.random.PRNGKey,
        dry_run: bool,
    ) -> Tuple[Any, Buffers, Dict[str, ObjData], Any, Dict[str, jnp.ndarray]]:
        pass

    @abc.abstractmethod
    def task_transition(
        self,
        actions: jnp.ndarray,
        state: PamyEnvState,
        new_buffers: Buffers,
        new_obj_data: Dict[str, ObjData],
        new_backend_state: Any,
        rng: jax.random.PRNGKey,
        **kwargs,
    ) -> Any:
        pass

    @abc.abstractmethod
    def task_observation(
        self,
        state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def task_reward(
        self,
        state: PamyEnvState,
        action: jnp.ndarray,
        next_state: PamyEnvState,
        rng: jax.random.PRNGKey,
        params: None = None,
    ) -> jnp.ndarray:
        pass

    def task_terminal(self, next_state: PamyEnvState) -> jnp.ndarray:
        return jnp.zeros(self.num_instances, dtype=bool)

    @abc.abstractmethod
    def task_episode_metrics(self, state: PamyEnvState) -> Dict[str, Any]:
        pass
