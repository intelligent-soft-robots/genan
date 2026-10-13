import functools
import logging
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import math
import numpy as np
import omegaconf
import optax
from flax import nnx
from tqdm import tqdm

from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.loss import ANLoss
from sim2real.actuator_network.models import (
    ActuatorEnsemble,
    restore_actuator_net_checkpoint,
    save_actuator_net_checkpoint,
)
from sim2real.actuator_network.simulation_accuracy_evaluator import (
    SimulationAccuracyEvaluator,
)
from sim2real.actuator_network.tensorboard_logs import TensorboardLogs
from sim2real.data_processing.process_data import build_dataset, process_data
from sim2real.util.config_utils import get_an_training_config

logger = logging.getLogger(__name__)


def split_datasets(
    cfg: omegaconf.DictConfig,
    data: Dataset,
    dt: float,
    train_split: float = 0.8,
) -> Tuple[
    Dataset,
    Dataset,
    Dict[str, jnp.ndarray],
]:
    num_train_trajs = math.floor(train_split * data.num_trajectories)
    # Use a fixed seed here to make sure that the train / validation split is the same during training and evaluation.
    mix_indices = np.random.default_rng(seed=42).permutation(data.num_trajectories)

    train_data = data[mix_indices[:num_train_trajs]]
    val_data = data[mix_indices[num_train_trajs:]]
    train_input, train_output = build_dataset(
        train_data.positions,
        train_data.observed_pressures,
        train_data.controls,
        train_data.torques,
        cfg.actuator_net,
        dt,
    )

    train_input_mean = jnp.array(np.mean(train_input, axis=0))
    train_input_std = jnp.array(np.std(train_input, axis=0))
    train_output_mean = jnp.array(np.mean(train_output, axis=0))
    train_output_std = jnp.array(np.std(train_output, axis=0))
    pos_diff = train_data.positions[:, 1:] - train_data.positions[:, :-1]
    pos_diff_mean = jnp.array(np.mean(pos_diff, axis=(0, 1)))
    pos_diff_std = jnp.array(np.std(pos_diff, axis=(0, 1)))
    pos_diff_min = pos_diff.min(axis=(0, 1))
    pos_diff_max = pos_diff.max(axis=(0, 1))
    acc = (
        train_data.positions[:, 2:]
        - 2 * train_data.positions[:, 1:-1]
        + train_data.positions[:, :-2]
    ) / dt**2
    acc_min = acc.min(axis=(0, 1))
    acc_max = acc.max(axis=(0, 1))
    train_data_stats = {
        "input_mean": train_input_mean,
        "input_std": train_input_std,
        "output_mean": train_output_mean,
        "output_std": train_output_std,
        "pos_diff_mean": pos_diff_mean,
        "pos_diff_std": pos_diff_std,
        "pos_diff_min": pos_diff_min,
        "pos_diff_max": pos_diff_max,
        "acc_min": acc_min,
        "acc_max": acc_max,
    }

    return train_data, val_data, train_data_stats


@functools.partial(nnx.jit, static_argnames=("loss",))
def train_epoch(rngs: nnx.Rngs, loss: ANLoss, optimizer: nnx.Optimizer) -> jnp.ndarray:
    optimizer.model.train()
    permutation = loss.get_permutation(rngs)

    def train_step(
        step: int, carry: Tuple[nnx.Optimizer, jnp.ndarray]
    ) -> Tuple[nnx.Optimizer, jnp.ndarray]:
        _optimizer, _sum_loss_models = carry
        (_, loss_models), grads = nnx.value_and_grad(loss, has_aux=True)(
            _optimizer.model, permutation, step
        )
        _optimizer.update(grads)
        return _optimizer, _sum_loss_models + loss_models

    _opt, sum_loss_models = nnx.fori_loop(
        0,
        loss.num_batches,
        train_step,
        (optimizer, jnp.zeros(optimizer.model.ensemble_size, dtype=jnp.float32)),
    )
    return sum_loss_models / loss.num_batches


def train_actuator_network(
    rngs: nnx.Rngs,
    cfg: omegaconf.DictConfig,
    model: ActuatorEnsemble,
    train_data: Dataset,
    val_data: Dataset,
    extra_eval_data: Optional[Dataset],
    save_models: bool = True,
) -> Dict[str, float | np.ndarray]:
    dt = 1.0 / cfg.target_frequency
    sim_acc_evaluator = SimulationAccuracyEvaluator(
        rngs,
        train_data,
        val_data,
        extra_eval_data,
        dt,
        cfg.actuator_net,
        cfg.end_effector,
        num_train_episodes=cfg.sim_acc_eval_episodes,
        num_val_episodes=cfg.sim_acc_eval_episodes,
        num_steps_per_episode=500,
    )

    train_data_in, train_data_torques = build_dataset(
        train_data.positions,
        train_data.observed_pressures,
        train_data.controls,
        train_data.torques,
        cfg.actuator_net,
        dt,
    )
    val_data_in, val_data_torques = build_dataset(
        val_data.positions,
        val_data.observed_pressures,
        val_data.controls,
        val_data.torques,
        cfg.actuator_net,
        dt,
    )

    if extra_eval_data is not None:
        extra_eval_data_in, extra_eval_data_torques = build_dataset(
            extra_eval_data.positions,
            extra_eval_data.observed_pressures,
            extra_eval_data.controls,
            extra_eval_data.torques,
            cfg.actuator_net,
            dt,
        )
    else:
        extra_eval_data_in = None
        extra_eval_data_torques = None

    tensorboard_logs = TensorboardLogs(
        Path.cwd(),
        jnp.array(train_data_in),
        jnp.array(train_data_torques),
        jnp.array(val_data_in),
        jnp.array(val_data_torques),
        jnp.array(extra_eval_data_in) if extra_eval_data_in is not None else None,
        (
            jnp.array(extra_eval_data_torques)
            if extra_eval_data_torques is not None
            else None
        ),
    )
    del (
        train_data_in,
        train_data_torques,
        val_data_in,
        val_data_torques,
        extra_eval_data_in,
        extra_eval_data_torques,
    )

    loss = hydra.utils.instantiate(
        cfg.loss,
        model.ensemble_size,
        train_data,
        val_data,
        extra_eval_data,
        cfg.actuator_net,
        dt,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay),
    )

    val_loss = np.array(loss.val_loss(model))
    tensorboard_logs.update_loss(None, val_loss, None, 0)
    tensorboard_logs.update_metrics(model, 0)
    logger.info(f"Initial validation loss: {val_loss.mean():.6}")
    _, min_val_model_state = nnx.split(model)
    min_val_losses = val_loss
    if cfg.sim_acc_eval_interval is not None:
        (
            (train_pos_errs, train_pos_errs_quant, train_pos_errs_filt, _),
            (val_pos_errs, val_pos_errs_quant, val_pos_errs_filt, _),
            (
                extra_eval_pos_errs,
                extra_eval_pos_errs_quant,
                extra_eval_pos_errs_filt,
                _,
            ),
        ) = sim_acc_evaluator.evaluate(model)
        tensorboard_logs.update_sim_error(
            train_pos_errs,
            val_pos_errs,
            extra_eval_pos_errs,
            train_pos_errs_quant,
            val_pos_errs_quant,
            extra_eval_pos_errs_quant,
            train_pos_errs_filt,
            val_pos_errs_filt,
            extra_eval_pos_errs_filt,
            dt,
            0,
        )
    for epoch in tqdm(range(1, cfg.num_epochs + 1)):
        train_loss = np.array(train_epoch(rngs, loss, optimizer))
        assert not np.any(np.isnan(train_loss))

        if epoch % cfg.log_interval == 0:
            logger.info(f"Epoch {epoch} Loss: {train_loss.mean():.6}")

        if epoch % cfg.val_interval == 0:
            val_loss = np.array(loss.val_loss(model))
            min_val_model_state_updated = False
            for i in range(model.ensemble_size):
                if val_loss[i] < min_val_losses[i]:
                    min_val_losses[i] = val_loss[i]
                    _, model_state = nnx.split(model)
                    model_state.update(
                        jax.tree.map(
                            lambda x, y: x.at[i].set(y[i]),
                            min_val_model_state.layers,
                            model_state.layers,
                        )
                    )
                    min_val_model_state_updated = True
            if save_models and min_val_model_state_updated:
                save_actuator_net_checkpoint(model, Path.cwd() / f"model_epoch_{epoch}")
                model_symlink = Path("model")
                if model_symlink.exists():
                    model_symlink.unlink()
                model_symlink.symlink_to(f"model_epoch_{epoch}")

            tensorboard_logs.update_metrics(model, epoch)
            logger.info(
                f"Validation loss: {val_loss.mean():.6}, minimum validation loss so far: {min_val_losses.mean():.6}"
            )
            extra_eval_loss = (
                np.array(loss.extra_eval_loss(model))
                if extra_eval_data is not None
                else None
            )
        else:
            val_loss = None
            extra_eval_loss = None
        tensorboard_logs.update_loss(train_loss, val_loss, extra_eval_loss, epoch)

        if (
            cfg.sim_acc_eval_interval is not None
            and epoch % cfg.sim_acc_eval_interval == 0
        ):
            (
                (train_pos_errs, train_pos_errs_quant, train_pos_errs_filt, _),
                (val_pos_errs, val_pos_errs_quant, val_pos_errs_filt, _),
                (
                    extra_eval_pos_errs,
                    extra_eval_pos_errs_quant,
                    extra_eval_pos_errs_filt,
                    _,
                ),
            ) = sim_acc_evaluator.evaluate(model)
            tensorboard_logs.update_sim_error(
                train_pos_errs,
                val_pos_errs,
                extra_eval_pos_errs,
                train_pos_errs_quant,
                val_pos_errs_quant,
                extra_eval_pos_errs_quant,
                train_pos_errs_filt,
                val_pos_errs_filt,
                extra_eval_pos_errs_filt,
                dt,
                epoch,
            )

    logger.info("Finished training actuator network")
    if save_models:
        # TODO: Use symlink here if the model was already saved
        save_actuator_net_checkpoint(model, Path.cwd() / f"final_model")
        _, state = nnx.split(model)
        logger.info("Finished saving final model")

    graphdef, _ = nnx.split(model)
    min_val_model = nnx.merge(graphdef, min_val_model_state)
    (
        (train_pos_errs, train_pos_errs_quant, train_pos_errs_filt, _),
        (val_pos_errs, val_pos_errs_quant, val_pos_errs_filt, _),
        (extra_eval_pos_errs, extra_eval_pos_errs_quant, extra_eval_pos_errs_filt, _),
    ) = sim_acc_evaluator.evaluate(min_val_model)
    logger.info("Finished evaluating best model")

    results = {
        "val_loss": val_loss.mean(),
        "train_pos_errs": np.array(jnp.abs(train_pos_errs).mean(0)),
        "train_pos_errs_quant": np.array(jnp.abs(train_pos_errs_quant).mean(0)),
        "train_pos_errs_filt": np.array(jnp.abs(train_pos_errs_filt).mean(0)),
        "val_pos_errs": np.array(jnp.abs(val_pos_errs).mean(0)),
        "val_pos_errs_quant": np.array(jnp.abs(val_pos_errs_quant).mean(0)),
        "val_pos_errs_filt": np.array(jnp.abs(val_pos_errs_filt).mean(0)),
    }
    if extra_eval_pos_errs is not None:
        results["extra_eval_pos_errs"] = np.array(jnp.abs(extra_eval_pos_errs).mean(0))
        results["extra_eval_pos_errs_quant"] = np.array(
            jnp.abs(extra_eval_pos_errs_quant).mean(0)
        )
        results["extra_eval_pos_errs_filt"] = np.array(
            jnp.abs(extra_eval_pos_errs_filt).mean(0)
        )
    logger.info(
        f"Best model validation: loss: {results['val_loss']:.4f}, "
        f"position error (first step): {results['val_pos_errs'][0].mean() * 180.0 / np.pi:.4f}°,"
        f"position error (last step): {results['val_pos_errs'][-1].mean() * 180.0 / np.pi:.2f}°."
    )

    return results


def create_and_train_actuator_network(
    cfg: omegaconf.DictConfig,
) -> Dict[str, float | np.ndarray]:
    if cfg.seed is not None:
        seed = cfg["seed"]
    else:
        seed = np.random.randint(0, 2**32 - 1)

    np.random.seed(seed)
    rngs = nnx.Rngs(params=seed + 1, backend=seed + 2, train=seed + 3)

    raw_data_path = Path(cfg.raw_data_path)
    if not raw_data_path.is_absolute():
        raw_data_path = Path(__file__).parents[2] / raw_data_path
    with raw_data_path.open("rb") as f:
        raw_data = pickle.load(f)

    raw_data = raw_data[cfg.num_warmup_trajectories :]

    if cfg.max_num_trajectories is not None:
        raw_data = raw_data[: cfg.max_num_trajectories]

    positions, positions_raw, observed_pressures, controls, torques = process_data(
        raw_data,
        cfg.end_effector,
        cfg.filter,
        cfg.target_frequency,
        cfg.actuation_delay,
        plot_visualizations=cfg.plot_dataset_visualizations,
    )

    dataset = Dataset(
        positions,
        positions_raw,
        observed_pressures,
        controls,
        torques,
    )

    train_data, val_data, train_data_stats = split_datasets(
        cfg, dataset, 1 / cfg.target_frequency
    )
    if cfg.extra_eval_exp_name is not None:
        exps_path = Path.cwd().parents[2]
        extra_eval_data_path = exps_path / cfg.extra_eval_exp_name / "raw_data.pkl"
        with extra_eval_data_path.open("rb") as f:
            raw_extra_eval_data = pickle.load(f)
        if cfg.max_num_trajectories is not None:
            raw_extra_eval_data = raw_extra_eval_data[: cfg.max_num_trajectories]

        (
            positions_extra,
            positions_raw_extra,
            observed_pressures_extra,
            controls_extra,
            torques_extra,
        ) = process_data(
            raw_extra_eval_data,
            cfg.end_effector,
            cfg.filter,
            cfg.target_frequency,
            cfg.actuation_delay,
            plot_visualizations=cfg.plot_dataset_visualizations,
        )
        extra_eval_data = Dataset(
            positions_extra,
            positions_raw_extra,
            observed_pressures_extra,
            controls_extra,
            torques_extra,
        )
    else:
        extra_eval_data = None

    if "init_checkpoint_an_name" in cfg and cfg.init_checkpoint_an_name is not None:
        init_checkpoint_exp_name = (
            cfg.init_checkpoint_exp_name
            if cfg.init_checkpoint_exp_name
            else cfg.exp_name
        )
        init_an_cfg = get_an_training_config(
            init_checkpoint_exp_name, cfg.init_checkpoint_an_name
        )
        if (
            init_an_cfg.actuator_net.ensemble_size != cfg.actuator_net.ensemble_size
            or init_an_cfg.actuator_net.num_hidden_layers
            != cfg.actuator_net.num_hidden_layers
            or init_an_cfg.actuator_net.num_neurons != cfg.actuator_net.num_neurons
            or init_an_cfg.actuator_net.activation_fn != cfg.actuator_net.activation_fn
        ):
            logger.warning(
                f"Network architecture of init checkpoint does not match current config."
                f"Using init checkpoint architecture instead."
            )
        model = restore_actuator_net_checkpoint(
            init_checkpoint_exp_name, cfg.init_checkpoint_an_name
        )
    else:
        model = hydra.utils.instantiate(cfg.actuator_net, rngs=rngs, **train_data_stats)

    results = train_actuator_network(
        rngs,
        cfg,
        model,
        train_data,
        val_data,
        extra_eval_data,
        cfg.save_models,
    )

    np.savez("results.npz", **results)

    return results


@hydra.main(
    version_base=None, config_path="config", config_name="train_actuator_network"
)
def main(cfg: omegaconf.DictConfig) -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")

    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        logger.info(f"Git commit: {sha}")
    except subprocess.CalledProcessError:
        logger.info("Not in a git repository")
    create_and_train_actuator_network(cfg)

    # Sometimes jobs on the cluster don't terminate on their own
    # --> force quit after waiting some time to complete background threads
    time.sleep(10)
    os._exit(0)


if __name__ == "__main__":
    main()
