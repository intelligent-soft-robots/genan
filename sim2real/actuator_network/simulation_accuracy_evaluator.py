import logging
from typing import Any, Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
from flax import nnx

from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.models import ActuatorEnsemble
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend
from sim2real.util.encoder_simulation import ENCODER_RESOLUTION

logger = logging.getLogger(__name__)


def err_quant(positions_sim: jax.Array, positions_gt: jax.Array) -> jax.Array:
    return jnp.where(
        positions_sim >= positions_gt,
        jnp.clip(positions_sim - positions_gt - 0.5 * ENCODER_RESOLUTION, min=0.0),
        jnp.clip(positions_sim - positions_gt + 0.5 * ENCODER_RESOLUTION, max=0.0),
    )


class SimulationAccuracyEvaluator:
    def __init__(
        self,
        rngs: nnx.Rngs,
        train_data: Dataset,
        val_data: Dataset,
        extra_eval_data: Optional[Dataset],
        actuator_net_timestep: float,
        actuator_net_cfg: Dict[str, Any] | omegaconf.DictConfig,
        end_effector: Literal["racket", "cup"],
        num_train_episodes: int = 200,
        num_val_episodes: int = 200,
        num_steps_per_episode: int = 100,
        step_all_models: bool = False,
    ):
        assert train_data.controls.shape[-2] == train_data.positions.shape[-2] - 1
        assert val_data.controls.shape[-2] == val_data.positions.shape[-2] - 1
        assert end_effector in [
            "racket",
            "cup",
        ], f"Invalid end effector: {end_effector}."
        # TODO: Use the hydra config to construct the env correctly
        self.backend = MjxBackend(
            context=end_effector,
            actuator_net_cfg=actuator_net_cfg,
            actuator_net_timestep=actuator_net_timestep,
            step_all_models=step_all_models,
            # TODO: Switch to single precision here
            simulation_precision="double",
        )
        self.rngs = rngs
        self.hist_len = self.backend.history_length

        self.num_train_episodes = num_train_episodes
        self.num_val_episodes = num_val_episodes

        if self.num_train_episodes > train_data.num_trajectories:
            self.num_train_episodes = train_data.num_trajectories
            logger.warning(
                f"Dataset does not contain enough training episodes. "
                f"Setting num_train_episodes to {train_data.num_trajectories}."
            )
        if self.num_val_episodes > val_data.num_trajectories:
            self.num_val_episodes = val_data.num_trajectories
            logger.warning(
                f"Dataset does not contain enough validation episodes. "
                f"Setting num_val_episodes to {val_data.num_trajectories}."
            )

        train_episode_indices = np.random.choice(
            np.arange(train_data.num_trajectories, dtype=int),
            self.num_train_episodes,
            replace=False,
        )
        train_data = train_data[train_episode_indices]
        val_episode_indices = np.random.choice(
            np.arange(val_data.num_trajectories, dtype=int),
            self.num_val_episodes,
            replace=False,
        )
        val_data = val_data[val_episode_indices]

        total_length = min(
            train_data.trajectory_length, num_steps_per_episode + self.hist_len + 1
        )

        positions = [
            train_data.positions,
            val_data.positions,
        ]
        positions_raw = [
            train_data.positions_raw,
            val_data.positions_raw,
        ]
        controls = [train_data.controls, val_data.controls]
        if extra_eval_data is not None:
            extra_eval_episode_indices = np.random.choice(
                np.arange(extra_eval_data.num_trajectories, dtype=int),
                self.num_val_episodes,
                replace=False,
            )
            extra_eval_data = extra_eval_data[extra_eval_episode_indices]
            positions.append(extra_eval_data.positions)
            controls.append(extra_eval_data.controls)
        self.positions = jnp.array(np.concatenate(positions)[:, -total_length:])
        if train_data.positions_raw is not None:
            self.positions_raw = jnp.array(
                np.concatenate(positions_raw)[:, -total_length:]
            )
        else:
            self.positions_raw = None
        self.controls = jnp.array(np.concatenate(controls)[:, -total_length + 1 :])

    def evaluate(self, actuator_ensemble: ActuatorEnsemble) -> Tuple[
        Tuple[jax.Array, jax.Array, jax.Array, np.ndarray],
        Tuple[jax.Array, jax.Array, jax.Array, np.ndarray],
        Tuple[
            Optional[jax.Array],
            Optional[jax.Array],
            Optional[jax.Array],
            Optional[np.ndarray],
        ],
    ]:
        actuator_ensemble.eval()
        if self.backend.step_all_models:
            pos_reset = self.positions[..., : (self.hist_len + 1), :][None].repeat(
                actuator_ensemble.ensemble_size, axis=0
            )
            ctrl_reset = self.controls[..., 1 : self.hist_len, :][None].repeat(
                actuator_ensemble.ensemble_size, axis=0
            )
            ctrl_traj = self.controls[..., self.hist_len :, :][None].repeat(
                actuator_ensemble.ensemble_size, axis=0
            )
        else:
            pos_reset = self.positions[:, : (self.hist_len + 1)]
            ctrl_reset = self.controls[:, 1 : self.hist_len]
            ctrl_traj = self.controls[:, self.hist_len :]
        positions_sim, _, info = self.backend.episode(
            self.rngs.backend(),
            pos_reset,
            ctrl_reset,
            ctrl_traj,
            actuator_ensemble=actuator_ensemble,
        )

        if self.backend.step_all_models:
            positions_sim = jnp.moveaxis(positions_sim, 0, 1)
            positions_gt_raw = self.positions_raw[:, None, (self.hist_len + 1) :]
            positions_gt_filtered = self.positions[:, None, (self.hist_len + 1) :]
        else:
            positions_gt_raw = self.positions_raw[:, (self.hist_len + 1) :]
            positions_gt_filtered = self.positions[:, (self.hist_len + 1) :]

        disagreements_sim = info["actuator_network"]["output"].std(axis=0)

        val_end = self.num_train_episodes + self.num_val_episodes

        train_results = (
            positions_sim[: self.num_train_episodes]
            - positions_gt_raw[: self.num_train_episodes],
            err_quant(
                positions_sim[: self.num_train_episodes],
                positions_gt_raw[: self.num_train_episodes],
            ),
            positions_sim[: self.num_train_episodes]
            - positions_gt_filtered[: self.num_train_episodes],
            disagreements_sim[: self.num_train_episodes],
        )
        val_results = (
            positions_sim[self.num_train_episodes : val_end]
            - positions_gt_raw[self.num_train_episodes : val_end],
            err_quant(
                positions_sim[self.num_train_episodes : val_end],
                positions_gt_raw[self.num_train_episodes : val_end],
            ),
            positions_sim[self.num_train_episodes : val_end]
            - positions_gt_filtered[self.num_train_episodes : val_end],
            disagreements_sim[self.num_train_episodes : val_end],
        )
        if positions_sim.shape[0] > val_end:
            extra_eval_results = (
                positions_sim[val_end:] - positions_gt_raw[val_end:],
                err_quant(positions_sim[val_end:], positions_gt_raw[val_end:]),
                positions_sim[val_end:] - positions_gt_filtered[val_end:],
                disagreements_sim[val_end:],
            )
        else:
            extra_eval_results = (None, None, None, None)

        return train_results, val_results, extra_eval_results
