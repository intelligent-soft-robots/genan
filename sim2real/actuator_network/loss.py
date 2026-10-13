import abc
import functools
from typing import Any, Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import optax
from flax import nnx
from tqdm import trange

from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.models import ActuatorEnsemble
from sim2real.actuator_network.standardize import standardize
from sim2real.data_processing.finite_differences import central_differences_second
from sim2real.data_processing.process_data import build_dataset, pad_history_length
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend


def get_histories(
    dataset: Dataset,
    episode_indices: jnp.ndarray,
    start_time_indices: jnp.ndarray,
    reset_hist_len: int,
    rollout_len: int,
) -> Dataset:
    time_indices_ctrl = start_time_indices[..., None] + jnp.arange(
        reset_hist_len + rollout_len
    )
    time_indices = start_time_indices[..., None] + jnp.arange(
        reset_hist_len + rollout_len + 1
    )
    return Dataset(
        dataset.positions[episode_indices[..., None], time_indices],
        (
            dataset.positions_raw[episode_indices[..., None], time_indices]
            if dataset.positions_raw is not None
            else None
        ),
        (
            dataset.observed_pressures[episode_indices[..., None], time_indices]
            if dataset.observed_pressures is not None
            else None
        ),
        dataset.controls[episode_indices[..., None], time_indices_ctrl],
        None,
    )


class ANLoss(abc.ABC):
    def __init__(
        self,
        ensemble_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_batches: int,
        bootstrapping: bool = False,
    ):
        self.ensemble_size = ensemble_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_batches = num_batches
        self.bootstrapping = bootstrapping

    @abc.abstractmethod
    def get_permutation(self, rngs: nnx.Rngs) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def __call__(
        self, model: ActuatorEnsemble, permutation: jnp.ndarray, batch_idx: int
    ) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def val_loss(self, model: ActuatorEnsemble) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def extra_eval_loss(self, model: ActuatorEnsemble) -> Optional[jnp.ndarray]:
        pass


class TorquePredictionLoss(ANLoss):

    def __init__(
        self,
        ensemble_size: int,
        train_data: Dataset,
        val_data: Dataset,
        extra_eval_data: Optional[Dataset],
        actuator_net_cfg: omegaconf.DictConfig | Dict[str, Any],
        actuator_net_timestep: float,
        train_batch_size: int,
        eval_batch_size: int,
        huber_delta: Optional[float],
        max_episode_length_s: Optional[float],
        error_scaling_factor: float = 1.0,
        l1_normalization: bool = False,
        bootstrapping: bool = False,
    ):
        if max_episode_length_s is not None:
            max_episode_length = round(max_episode_length_s / actuator_net_timestep)
            train_data = Dataset(
                train_data.positions[:, : max_episode_length + 1],
                train_data.positions_raw[:, : max_episode_length + 1],
                train_data.observed_pressures[:, : max_episode_length + 1],
                train_data.controls[:, :max_episode_length],
                train_data.torques[:, : max_episode_length + 1],
            )
            val_data = Dataset(
                val_data.positions[:, : max_episode_length + 1],
                val_data.positions_raw[:, : max_episode_length + 1],
                val_data.observed_pressures[:, : max_episode_length + 1],
                val_data.controls[:, :max_episode_length],
                val_data.torques[:, : max_episode_length + 1],
            )
        train_data_in, train_data_out = build_dataset(
            train_data.positions,
            train_data.observed_pressures,
            train_data.controls,
            train_data.torques,
            actuator_net_cfg,
            actuator_net_timestep,
        )
        self.train_data_in = jnp.array(train_data_in)
        self.train_data_out = jnp.array(train_data_out)
        val_data_in, val_data_out = build_dataset(
            val_data.positions,
            val_data.observed_pressures,
            val_data.controls,
            val_data.torques,
            actuator_net_cfg,
            actuator_net_timestep,
        )
        self.val_data_in = jnp.array(val_data_in)
        self.val_data_out = jnp.array(val_data_out)
        if extra_eval_data is not None:
            extra_eval_data_in, extra_eval_data_out = build_dataset(
                extra_eval_data.positions,
                extra_eval_data.observed_pressures,
                extra_eval_data.controls,
                extra_eval_data.torques,
                actuator_net_cfg,
                actuator_net_timestep,
            )
            self.extra_eval_data_in = jnp.array(extra_eval_data_in)
            self.extra_eval_data_out = jnp.array(extra_eval_data_out)
        else:
            self.extra_eval_data_in = None
            self.extra_eval_data_out = None
        self.huber_delta = huber_delta
        self.error_scaling_factor = error_scaling_factor

        self.l1_normalization = l1_normalization
        self.train_data_abs_mean = jnp.abs(self.train_data_out).mean(axis=0)

        num_train_batches = train_data_in.shape[0] // train_batch_size
        self._num_train_samples = train_data_in.shape[0]
        super().__init__(
            ensemble_size, train_batch_size, eval_batch_size, num_train_batches, bootstrapping
        )

    def get_permutation(self, rngs: nnx.Rngs) -> jnp.ndarray:
        num_total = self.num_batches * self.train_batch_size
        return jnp.stack(
            [
                (
                    jax.random.choice(
                        rngs.train(), self._num_train_samples, shape=(num_total,), replace=True
                    )
                    if self.bootstrapping
                    else jax.random.permutation(rngs.train(), num_total)
                ).reshape((self.num_batches, self.train_batch_size))
                for _ in range(self.ensemble_size)
            ],
            axis=1,
        )

    def __call__(
        self, model: ActuatorEnsemble, permutation: jnp.ndarray, batch_idx: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_in = self.train_data_in[permutation[batch_idx]]
        batch_out = self.train_data_out[permutation[batch_idx]]
        loss, loss_models = self.compute_loss(
            model,
            batch_in,
            batch_out,
            self.huber_delta,
            self.error_scaling_factor,
            self.l1_normalization,
            self.train_data_abs_mean,
        )
        return loss, loss_models.mean(axis=1)

    @classmethod
    @functools.partial(
        nnx.jit,
        static_argnames=(
            "cls",
            "huber_delta",
            "error_scaling_factor",
            "l1_normalization",
        ),
    )
    def compute_loss(
        cls,
        model: ActuatorEnsemble,
        batch_in: jnp.ndarray,
        batch_out: jnp.ndarray,
        huber_delta: float,
        error_scaling_factor: float,
        l1_normalization: bool,
        train_data_abs_mean: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        predicted_output_destd, predicted_output, _ = model(batch_in, None)
        if l1_normalization:
            loss_models = (
                (
                    error_scaling_factor
                    * (predicted_output_destd - batch_out)
                    / train_data_abs_mean
                )
                ** 2
            ).mean(axis=2)
        else:
            batch_out_std = standardize(batch_out, model.output_mean, model.output_std)
            if huber_delta is None:
                loss_models = (
                    (error_scaling_factor * (predicted_output - batch_out_std)) ** 2
                ).mean(axis=2)
            else:
                loss_models = optax.losses.huber_loss(
                    predicted_output, batch_out_std, huber_delta
                ).mean(axis=2)
        loss = loss_models.mean()
        return loss, loss_models

    def val_loss(self, model: ActuatorEnsemble) -> jnp.ndarray:
        return self.evaluate(
            model,
            self.val_data_in,
            self.val_data_out,
            self.eval_batch_size,
            self.ensemble_size,
            self.huber_delta,
            self.error_scaling_factor,
            self.l1_normalization,
            self.train_data_abs_mean,
        )

    def extra_eval_loss(self, model: ActuatorEnsemble) -> Optional[jnp.ndarray]:
        if self.extra_eval_data_in is not None:
            return self.evaluate(
                model,
                self.extra_eval_data_in,
                self.extra_eval_data_out,
                self.eval_batch_size,
                self.ensemble_size,
                self.huber_delta,
                self.error_scaling_factor,
                self.l1_normalization,
                self.train_data_abs_mean,
            )
        else:
            return None

    @classmethod
    @functools.partial(
        nnx.jit,
        static_argnames=(
            "cls",
            "eval_batch_size",
            "ensemble_size",
            "huber_delta",
            "error_scaling_factor",
            "l1_normalization",
        ),
    )
    def evaluate(
        cls,
        model: ActuatorEnsemble,
        data_in: jnp.ndarray,
        data_out: jnp.ndarray,
        eval_batch_size: int,
        ensemble_size: int,
        huber_delta: float,
        error_scaling_factor: float,
        l1_normalization: bool,
        train_data_abs_mean: jnp.ndarray,
    ) -> jnp.ndarray:
        num_samples = data_in.shape[0]
        num_batches = (num_samples + eval_batch_size - 1) // eval_batch_size
        padding_width = num_batches * eval_batch_size - num_samples
        data_in = jnp.pad(data_in, ((0, padding_width), (0, 0)))
        data_out = jnp.pad(data_out, ((0, padding_width), (0, 0)))

        batches_in = data_in.reshape(num_batches, eval_batch_size, -1)
        batches_out = data_out.reshape(num_batches, eval_batch_size, -1)

        def batch_loss_fn(
            batch_idx: int, carry: Tuple[ActuatorEnsemble, jnp.ndarray]
        ) -> Tuple[ActuatorEnsemble, jnp.ndarray]:
            _model, _losses = carry
            val_batch_in = batches_in[batch_idx][None].repeat(ensemble_size, axis=0)
            val_batch_out = batches_out[batch_idx][None].repeat(ensemble_size, axis=0)
            batch_loss = cls.compute_loss(
                _model,
                val_batch_in,
                val_batch_out,
                huber_delta,
                error_scaling_factor,
                l1_normalization,
                train_data_abs_mean,
            )[1]
            return _model, _losses.at[batch_idx].set(batch_loss)

        _, losses = nnx.fori_loop(
            0,
            num_batches,
            batch_loss_fn,
            (
                model,
                jnp.zeros(
                    (num_batches, ensemble_size, eval_batch_size),
                    dtype=jnp.float32,
                ),
            ),
        )
        # Do not take loss of padding input into account
        losses = jnp.concatenate(losses, axis=1)[:, :num_samples]
        return losses.mean(axis=1)


class PositionAccuracyLoss(ANLoss):
    def __init__(
        self,
        ensemble_size: int,
        train_data: Dataset,
        val_data: Dataset,
        extra_eval_data: Optional[Dataset],
        actuator_net_cfg: omegaconf.DictConfig | Dict[str, Any],
        actuator_net_timestep: float,
        train_batch_size: int,
        eval_batch_size: int,
        rollout_len: int,
        incomplete_histories: bool,
        normalize_loss: bool,
        stop_gradients_an_input: bool,
        end_effector: Literal["racket", "cup"],
        bootstrapping: bool = False,
    ):
        self.mjx_backend = MjxBackend(
            context=end_effector,
            actuator_net_cfg=actuator_net_cfg,
            actuator_net_timestep=actuator_net_timestep,
            step_all_models=True,
            stop_gradients_an_input=stop_gradients_an_input,
            simulation_precision="double",
        )
        self.num_train_episodes = train_data.num_trajectories
        self.reset_hist_len = (
            actuator_net_cfg.num_history_states * actuator_net_cfg.step_size + 1
        )
        self.rollout_len = rollout_len

        if normalize_loss:
            self._loss_normalization = jnp.zeros(
                (self.rollout_len, 4), dtype=jnp.float32
            )
            # TODO: Parallelize multiple episodes (should use the same batch size as the evaluation)
            for ep in trange(
                train_data.num_trajectories, desc="Computing normalization"
            ):
                positions = jnp.array(train_data.positions[ep, 1 : -self.rollout_len])
                prev_positions = jnp.array(
                    train_data.positions[ep, : -(self.rollout_len + 1)]
                )
                _, _, state, _ = self.mjx_backend.reset(
                    jax.random.PRNGKey(42),  # Unused
                    jnp.zeros_like(positions),
                    positions,
                    prev_positions,
                    ensemble_size=1,
                )
                mjx_data = state.mjx_data
                next_positions = []
                for step in range(self.rollout_len):
                    mjx_data = self.mjx_backend.step_torque(
                        mjx_data, jnp.zeros_like(positions)
                    )
                    next_positions.append(mjx_data.qpos.astype(jnp.float32))
                next_positions_sim = jnp.stack(next_positions, axis=0)
                next_positions_gt = jnp.array(
                    np.stack(
                        [
                            train_data.positions[
                                ep,
                                (i + 2) : train_data.trajectory_length
                                - self.rollout_len
                                + i
                                + 1,
                            ]
                            for i in range(0, self.rollout_len)
                        ],
                        axis=0,
                    )
                )
                self._loss_normalization += jnp.abs(
                    next_positions_sim - next_positions_gt
                ).mean(axis=1)
            self._loss_normalization /= train_data.num_trajectories
        else:
            self._loss_normalization = jnp.ones(
                (self.rollout_len, 1), dtype=jnp.float32
            )

        if incomplete_histories:
            self.train_data = self._pad_dataset(
                train_data, self.reset_hist_len
            ).to_jax()
            self.val_data = self._pad_dataset(val_data, self.reset_hist_len).to_jax()
            if extra_eval_data is not None:
                self.extra_eval_data = self._pad_dataset(
                    extra_eval_data, self.reset_hist_len
                ).to_jax()
            else:
                self.extra_eval_data = None
        else:
            self.train_data = train_data.to_jax()
            self.val_data = val_data.to_jax()
            self.extra_eval_data = (
                extra_eval_data.to_jax() if extra_eval_data is not None else None
            )
        self.traj_len = self.train_data.trajectory_length
        num_batches = (
            self.num_train_episodes
            * (self.traj_len - self.reset_hist_len - self.rollout_len)
            // train_batch_size
        )
        self.stop_gradients_an_input = stop_gradients_an_input
        self._valid_starts_shape = (
            self.num_train_episodes,
            self.traj_len - self.reset_hist_len - self.rollout_len,
        )
        super().__init__(ensemble_size, train_batch_size, eval_batch_size, num_batches, bootstrapping)

    def get_permutation(self, rngs: nnx.Rngs) -> jnp.ndarray:
        all_indices = jnp.indices(self._valid_starts_shape).reshape(2, -1).T
        n_valid = all_indices.shape[0]
        num_total = self.num_batches * self.train_batch_size
        return jnp.stack(
            [
                (
                    all_indices[
                        jax.random.choice(rngs.train(), n_valid, shape=(num_total,), replace=True)
                    ]
                    if self.bootstrapping
                    else jax.random.permutation(rngs.train(), all_indices)[:num_total]
                ).reshape((self.num_batches, self.train_batch_size, 2))
                for _ in range(self.ensemble_size)
            ],
            axis=1,
        )

    def __call__(
        self, model: ActuatorEnsemble, permutation: jnp.ndarray, batch_idx: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_data = get_histories(
            self.train_data,
            permutation[batch_idx, ..., 0],
            permutation[batch_idx, ..., 1],
            self.reset_hist_len,
            self.rollout_len,
        )
        loss, loss_models = self.compute_loss(
            model,
            batch_data.positions,
            batch_data.controls,
            self._loss_normalization,
            self.mjx_backend,
            self.reset_hist_len,
        )
        return loss, loss_models.mean(axis=1)

    @classmethod
    @functools.partial(
        nnx.jit, static_argnames=("cls", "mjx_backend", "reset_hist_len")
    )
    def compute_loss(
        cls,
        model: ActuatorEnsemble,
        position_histories: jnp.ndarray,
        control_histories: jnp.ndarray,
        loss_normalization: jnp.ndarray,
        mjx_backend: MjxBackend,
        reset_hist_len: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        positions_pred, _, _ = mjx_backend.episode(
            jax.random.PRNGKey(42),  # Unused
            position_histories[:, :, : reset_hist_len + 1],
            control_histories[:, :, 1:reset_hist_len],
            control_histories[:, :, reset_hist_len:],
            actuator_ensemble=model,
        )

        sim_err = positions_pred - position_histories[:, :, (reset_hist_len + 1) :]
        loss_models = ((sim_err / loss_normalization[None, None]) ** 2).mean(
            axis=(2, 3)
        )
        loss = loss_models.mean()
        return loss, loss_models

    def val_loss(self, model: ActuatorEnsemble) -> jnp.ndarray:
        indices = jnp.indices(
            (
                self.val_data.num_trajectories,
                self.traj_len - self.reset_hist_len - self.rollout_len,
            )
        )
        histories = get_histories(
            self.val_data,
            indices[0],
            indices[1],
            self.reset_hist_len,
            self.rollout_len,
        )
        return self.evaluate(
            model,
            histories.positions,
            histories.controls,
            self._loss_normalization,
            self.mjx_backend,
            self.ensemble_size,
            self.eval_batch_size,
            self.reset_hist_len,
        )

    def extra_eval_loss(self, model: ActuatorEnsemble) -> Optional[jnp.ndarray]:
        if self.extra_eval_data is not None:
            indices = jnp.indices(
                (
                    self.extra_eval_data.num_trajectories,
                    self.traj_len - self.reset_hist_len - self.rollout_len,
                )
            )
            histories = get_histories(
                self.extra_eval_data,
                indices[0],
                indices[1],
                self.reset_hist_len,
                self.rollout_len,
            )
            return self.evaluate(
                model,
                histories.positions,
                histories.controls,
                self._loss_normalization,
                self.mjx_backend,
                self.ensemble_size,
                self.eval_batch_size,
                self.reset_hist_len,
            )
        else:
            return None

    @classmethod
    @functools.partial(
        nnx.jit,
        static_argnames=(
            "cls",
            "mjx_backend",
            "ensemble_size",
            "eval_batch_size",
            "reset_hist_len",
        ),
    )
    def evaluate(
        cls,
        model: ActuatorEnsemble,
        position_histories: jnp.ndarray,
        control_histories: jnp.ndarray,
        loss_normalization: jnp.ndarray,
        mjx_backend: MjxBackend,
        ensemble_size: int,
        eval_batch_size: int,
        reset_hist_len: int,
    ) -> jnp.ndarray:
        # Merge data from different episodes
        position_histories = position_histories.reshape(
            -1, *position_histories.shape[2:]
        )
        control_histories = control_histories.reshape(-1, *control_histories.shape[2:])

        num_samples = position_histories.shape[0]
        num_batches = (num_samples + eval_batch_size - 1) // eval_batch_size
        padding_width = num_batches * eval_batch_size - num_samples
        position_histories = jnp.pad(
            position_histories, ((0, padding_width), (0, 0), (0, 0))
        )
        control_histories = jnp.pad(
            control_histories, ((0, padding_width), (0, 0), (0, 0))
        )
        position_batches = position_histories.reshape(
            num_batches, eval_batch_size, -1, 4
        )
        control_batches = control_histories.reshape(num_batches, eval_batch_size, -1, 4)

        def batch_loss_fn(
            batch_idx: int, carry: Tuple[ActuatorEnsemble, jnp.ndarray]
        ) -> Tuple[ActuatorEnsemble, jnp.ndarray]:
            _model, _losses = carry
            pos_batch = position_batches[batch_idx][None].repeat(ensemble_size, axis=0)
            control_batch = control_batches[batch_idx][None].repeat(
                ensemble_size, axis=0
            )
            batch_loss = cls.compute_loss(
                _model,
                pos_batch,
                control_batch,
                loss_normalization,
                mjx_backend,
                reset_hist_len,
            )[1]
            return _model, _losses.at[batch_idx].set(batch_loss)

        _, losses = nnx.fori_loop(
            0,
            num_batches,
            batch_loss_fn,
            (
                model,
                jnp.zeros(
                    (num_batches, ensemble_size, eval_batch_size),
                    dtype=jnp.float32,
                ),
            ),
        )
        # Do not take loss of padding input into account
        losses = jnp.concatenate(losses, axis=1)[:, :num_samples]
        return losses.mean(axis=1)

    @staticmethod
    def _pad_dataset(dataset: Dataset, hist_len: int) -> Dataset:
        positions, positions_raw, observed_pressures, controls = pad_history_length(
            dataset.positions,
            dataset.positions_raw,
            dataset.observed_pressures,
            dataset.controls,
            hist_len,
        )
        return Dataset(
            positions, positions_raw, observed_pressures, controls, dataset.torques
        )
