import functools
from typing import Any, Dict, Optional, Tuple

import gymnasium.experimental.functional
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
from flax import nnx
from jax.tree_util import register_pytree_node_class

from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.loss import get_histories
from sim2real.actuator_network.pack_an_input import pack_an_input
from sim2real.data_processing.process_data import pad_history_length
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend
from sim2real.rl_tasks.physics_backend.physics_backend import Buffers
from sim2real.util.config_utils import get_an_input_size_from_cfg


@register_pytree_node_class
class ActuatorNetEnvState:
    def __init__(
        self,
        buffers: Buffers,
        backend_state: Any,
        step: int,
        last_torque: jnp.ndarray,
        gt_pos_traj: jnp.ndarray,
        ctrl_traj: jnp.ndarray,
        episode_metrics: Dict[str, jnp.ndarray],
    ):
        self.buffers = buffers
        self.backend_state = backend_state
        self.step = step
        self.torque = last_torque
        self.gt_pos_traj = gt_pos_traj
        self.ctrl_traj = ctrl_traj
        self.episode_metrics = episode_metrics

    def tree_flatten(self):
        children = (
            self.buffers,
            self.backend_state,
            self.step,
            self.torque,
            self.gt_pos_traj,
            self.ctrl_traj,
            self.episode_metrics,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def init(
        cls,
        buffers: Buffers,
        backend_state: Any,
        gt_pos_traj: jnp.ndarray,
        ctrl_traj: jnp.ndarray,
        episode_metrics: Dict[str, jnp.ndarray],
    ) -> "ActuatorNetEnvState":
        num_instances = buffers.controls.shape[0]
        return cls(
            buffers,
            backend_state,
            0,
            jnp.nan * jnp.ones((num_instances, 4), dtype=jnp.float32),
            gt_pos_traj,
            ctrl_traj,
            episode_metrics,
        )

    def update(
        self,
        buffers: Buffers,
        backend_state: Any,
        torque: jnp.ndarray,
        curr_step_metrics: Dict[str, jnp.ndarray],
    ) -> "ActuatorNetEnvState":
        episode_metrics = {
            k: ep + curr_step_metrics[k] for k, ep in self.episode_metrics.items()
        }
        return ActuatorNetEnvState(
            buffers,
            backend_state,
            self.step + 1,
            torque,
            self.gt_pos_traj,
            self.ctrl_traj,
            episode_metrics,
        )


class ActuatorNetEnv(
    gymnasium.experimental.functional.FuncEnv[
        ActuatorNetEnvState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        None,
        Any,
    ],
):
    def __init__(
        self,
        backend: MjxBackend,
        num_instances: int,
        train_data: Dataset,
        actuator_net_cfg: omegaconf.DictConfig | Dict[str, Any],
        reward_scale_pos: float,
        reward_scale_smoothness: float,
        reward_pos_exponent: Optional[float],
        incomplete_histories: bool,
        rollout_len: int,
        gt_position_observation: bool = False,
    ):
        super().__init__()
        self.backend = backend
        self.num_instances = num_instances
        self.actuator_net_cfg = actuator_net_cfg
        self.gt_position_observation = gt_position_observation
        assert reward_scale_pos >= 0.0
        self.reward_scale_pos = reward_scale_pos
        assert reward_scale_smoothness >= 0.0
        self.reward_scale_smoothness = reward_scale_smoothness
        assert reward_pos_exponent is None or reward_pos_exponent > 0.0
        self.reward_pos_exponent = reward_pos_exponent

        # TODO: Set proper limits
        obs_size = get_an_input_size_from_cfg(actuator_net_cfg)
        if self.gt_position_observation:
            obs_size += 4
        obs_lim_low = -np.inf * np.ones(obs_size, dtype=np.float32)
        obs_lim_high = np.inf * np.ones(obs_size, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(obs_lim_low, obs_lim_high)
        self.observation_spaces = gymnasium.spaces.Box(
            obs_lim_low[None].repeat(self.num_instances, axis=0),
            obs_lim_high[None].repeat(self.num_instances, axis=0),
        )
        act_lim_low = -np.inf * np.ones(4, dtype=np.float32)
        act_lim_high = np.inf * np.ones(4, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(act_lim_low, act_lim_high)
        self.action_spaces = gymnasium.spaces.Box(
            act_lim_low[None].repeat(self.num_instances, axis=0),
            act_lim_high[None].repeat(self.num_instances, axis=0),
        )

        self.reset_hist_len = (
            actuator_net_cfg["num_history_states"] * actuator_net_cfg["step_size"] + 1
        )
        if incomplete_histories:
            positions, positions_raw, observed_pressures, controls = pad_history_length(
                train_data.positions,
                train_data.positions_raw,
                train_data.observed_pressures,
                train_data.controls,
                self.reset_hist_len,
            )
            self.train_data = Dataset(
                positions,
                positions_raw,
                observed_pressures,
                controls,
                train_data.torques,
            ).to_jax()
        else:
            self.train_data = train_data.to_jax()
        self.rollout_len = (
            rollout_len
            if rollout_len is not None and rollout_len > 0
            else self.train_data.controls.shape[1] - self.reset_hist_len - 1
        )

    def _pack_observation(self, buffers: Buffers, gt_pos: jnp.ndarray) -> jnp.ndarray:
        obs = pack_an_input(
            buffers.positions,
            None,
            buffers.controls,
            self.actuator_net_cfg,
            self.backend.dt,
        )
        if self.gt_position_observation:
            obs = jnp.concatenate((obs, gt_pos.astype(jnp.float32)), axis=-1)
        return obs

    @functools.partial(nnx.jit, static_argnames=("self",))
    def reset(self, rng: jax.random.PRNGKey) -> Tuple[
        Tuple[jnp.ndarray, Dict[str, Any]],
        ActuatorNetEnvState,
    ]:
        dataset_size, traj_len, num_dofs = self.train_data.positions.shape
        rng, rng_indices = jax.random.split(rng)
        valid_starts_shape = (
            dataset_size,
            traj_len - self.reset_hist_len - self.rollout_len,
        )
        indices = jax.random.choice(
            rng_indices,
            jnp.indices(valid_starts_shape).reshape(2, -1).T,
            (self.num_instances,),
        )
        trajs = get_histories(
            self.train_data,
            indices[:, 0],
            indices[:, 1],
            self.reset_hist_len,
            self.rollout_len,
        )
        gt_pos_traj = trajs.positions
        ctrl_traj = trajs.controls
        buffers, _, backend_state, _ = self.backend.reset_with_history(
            rng,
            gt_pos_traj[:, : self.reset_hist_len + 1],
            ctrl_traj[:, 1 : self.reset_hist_len],
            ensemble_size=1,
        )
        buffers = buffers.advance_ctrl(ctrl_traj[:, self.reset_hist_len])

        episode_metrics_init = {
            "rewards/reward_pos": jnp.array(0.0, dtype=jnp.float32),
            "rewards/reward_smoothness": jnp.array(0.0, dtype=jnp.float32),
        }

        state = ActuatorNetEnvState.init(
            buffers, backend_state, gt_pos_traj, ctrl_traj, episode_metrics_init
        )

        obs = self._pack_observation(buffers, gt_pos_traj[:, self.reset_hist_len])

        info = {"episode": episode_metrics_init}
        return (obs, info), state

    @functools.partial(nnx.jit, static_argnames=("self",))
    def step(
        self,
        action: jnp.ndarray,
        state: ActuatorNetEnvState,
        rng: jax.random.PRNGKey,
        **kwargs,
    ) -> Tuple[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]],
        ActuatorNetEnvState,
    ]:
        traj_step = state.step + self.reset_hist_len + 1

        buffers = state.buffers
        mjx_data = self.backend.step_torque(state.backend_state.mjx_data, action)
        backend_state = state.backend_state.update(mjx_data, None, None)
        buffers = buffers.advance_pos(backend_state.mjx_data.qpos.astype(jnp.float32))
        next_ctrl_idx = jnp.minimum(traj_step, state.ctrl_traj.shape[1] - 1)
        buffers = buffers.advance_ctrl(state.ctrl_traj[:, next_ctrl_idx])

        gt_pos = state.gt_pos_traj[:, traj_step]
        observation = self._pack_observation(buffers, gt_pos)
        pos_err = gt_pos - backend_state.mjx_data.qpos.astype(jnp.float32)
        if self.reward_pos_exponent is not None:
            reward_pos = self.reward_scale_pos * jnp.exp(
                -self.reward_pos_exponent * jnp.mean(pos_err**2, axis=-1)
            )
        else:
            reward_pos = -self.reward_scale_pos * jnp.mean(
                jnp.abs(pos_err),
                axis=-1,
            )
        # In the first step there is no previous action, set the penalty to zero
        reward_smoothness = self.reward_scale_smoothness * jnp.where(
            jnp.all(jnp.isnan(state.torque), axis=1),
            jnp.zeros(self.num_instances, dtype=jnp.float32),
            jnp.exp(-0.5 * jnp.mean(jnp.abs(action - state.torque), axis=-1)),
        )
        reward = reward_pos + reward_smoothness

        terminated = jnp.zeros(self.num_instances, dtype=bool)
        truncated = jnp.full(
            self.num_instances, traj_step >= state.gt_pos_traj.shape[1] - 1
        )
        episode_metrics = {
            "rewards/reward_pos": reward_pos.mean(),
            "rewards/reward_smoothness": reward_smoothness.mean(),
        }

        next_state = state.update(buffers, backend_state, action, episode_metrics)

        info = {
            "episode": {
                k: v / next_state.step for k, v in next_state.episode_metrics.items()
            }
        }
        return (observation, reward, terminated, truncated, info), next_state

    def close(self) -> None:
        return
