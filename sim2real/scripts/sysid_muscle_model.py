"""System identification for the analytic muscle model.

The physical parameters are fitted to the multi-step position error of a MJX rollout, which is
the quantity SimulationAccuracyEvaluator reports. The inverse-dynamics torque labels are
deliberately not used as a regression target: they are twice-differentiated from quantized
encoder positions and are noise-dominated on the proximal joints.

Run from the repository root, e.g.
    python -m sim2real.scripts.sysid_muscle_model exp_name=2026-01-22_black_racket \\
        run_name=analytic_v1 end_effector=racket
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import optax
from flax import nnx
from scipy.signal import lfilter
from tensorboardX import SummaryWriter
from tqdm import tqdm

from sim2real.actuator_network.analytic_muscle_model import (
    NUM_DOFS,
    AnalyticMuscleEnsemble,
)
from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.loss import PositionAccuracyLoss
from sim2real.actuator_network.models import save_actuator_net_checkpoint
from sim2real.actuator_network.simulation_accuracy_evaluator import (
    SimulationAccuracyEvaluator,
)
from sim2real.actuator_network.tensorboard_logs import SIM_EVAL_STEPS, log_sim_errors
from sim2real.data_processing.process_data import process_data

logger = logging.getLogger(__name__)

# Rollout length reported alongside the mean, in actuator-network steps (1 s at 500 Hz).
FULL_ROLLOUT_STEPS = 500


def check_dataset_consistency(
    actuator_net_cfg: omegaconf.DictConfig, exp_path: Path
) -> None:
    """Make sure the model's robot constants match the dataset it is identified on."""
    with (exp_path / "robot_config.json").open() as f:
        robot_config = json.load(f)
    for key in [
        "min_pressures_ago",
        "min_pressures_antago",
        "max_pressures_ago",
        "max_pressures_antago",
    ]:
        expected = list(robot_config[key])
        actual = list(actuator_net_cfg[key])
        assert (
            expected == actual
        ), f"{key} does not match the dataset, expected {expected}, got {actual}."
    collect_cfg = omegaconf.OmegaConf.load(
        exp_path / "collect_data_config" / "config.yaml"
    )
    expected_ndp = list(collect_cfg.ndp)
    actual_ndp = list(actuator_net_cfg.ndp)
    assert (
        expected_ndp == actual_ndp
    ), f"ndp does not match the dataset, expected {expected_ndp}, got {actual_ndp}."


def check_pull_signs(
    positions: np.ndarray, controls: np.ndarray, pull_signs: Sequence[float]
) -> None:
    """Verify that the antagonist of each pair really drives the joint in the assumed direction.

    Raising the control raises the antagonist pressure, so the correlation between control and
    joint angle has to carry the sign the model assumes. Getting this wrong makes the model
    unfittable rather than merely inaccurate, so it is checked rather than fitted.
    """
    for dof in range(NUM_DOFS):
        correlation = np.corrcoef(
            controls[:, :, dof].ravel(), positions[:, 1:, dof].ravel()
        )[0, 1]
        logger.info(f"DoF {dof}: corr(control, position) = {correlation:+.3f}")
        assert np.sign(correlation) == np.sign(pull_signs[dof]), (
            f"antagonist_pull_signs[{dof}] = {pull_signs[dof]} contradicts the data "
            f"(corr(control, position) = {correlation:+.3f})."
        )


def pressure_lag_explained_variance(
    desired_pressures: np.ndarray,
    observed_pressures: np.ndarray,
    timestep: float,
    time_constants: np.ndarray,
    warmup_steps: int = 300,
) -> np.ndarray:
    """Explained variance of a lagged desired pressure, shape [taus, dofs, muscles]."""
    scores = np.zeros((len(time_constants), NUM_DOFS, 2))
    for dof in range(NUM_DOFS):
        for muscle in range(2):
            desired = desired_pressures[:, :, dof, muscle].astype(np.float64)
            observed = observed_pressures[:, :, dof, muscle].astype(np.float64)
            target = observed[:, warmup_steps:]
            for i, time_constant in enumerate(time_constants):
                decay = np.exp(-timestep / time_constant)
                lagged = lfilter([1 - decay], [1, -decay], desired, axis=1)
                mse = ((lagged[:, warmup_steps:] - target) ** 2).mean()
                scores[i, dof, muscle] = 1.0 - mse / target.var()
    return scores


def calibrate_pressure_lag(
    desired_pressures: np.ndarray,
    observed_pressures: np.ndarray,
    timestep: float,
    time_constants: np.ndarray,
    shared: bool = True,
    warmup_steps: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid-fit a first-order lag to the recorded pressures.

    All eight muscles sit next to each other in the base and share a valve bank and supply, so by
    default a single time constant is fitted for all of them: the pressure dynamics cannot
    plausibly differ per joint, and anything that does differ per joint belongs downstream of the
    pressure (tendon elasticity, friction, inertia) rather than in this lag.

    Returns (time_constant, explained_variance), each of shape [dofs].
    """
    scores = pressure_lag_explained_variance(
        desired_pressures,
        observed_pressures,
        timestep,
        time_constants,
        warmup_steps,
    )
    best = scores.argmax(axis=0)
    for dof in range(NUM_DOFS):
        for muscle in range(2):
            logger.info(
                f"DoF {dof} muscle {muscle}: best fit time constant "
                f"{time_constants[best[dof, muscle]] * 1e3:.0f} ms, explained variance "
                f"{scores[best[dof, muscle], dof, muscle]:.3f}"
            )

    if not shared:
        weights = np.clip(scores.max(axis=0), 1e-3, None)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return (
            (time_constants[best] * weights).sum(axis=1),
            (scores.max(axis=0) * weights).sum(axis=1),
        )

    # Muscles whose commanded pressure barely varies (the antagonist of DoF 1 spans only 1400 of
    # the ~10000 units the others use) carry almost no information, so they are excluded from the
    # shared fit rather than allowed to drag it.
    informative = scores.max(axis=0) > 0.5
    pooled = scores[:, informative].mean(axis=-1)
    i = int(pooled.argmax())
    logger.info(
        f"Shared pressure lag over {informative.sum()} informative muscles: time constant "
        f"{time_constants[i] * 1e3:.0f} ms, mean explained variance {pooled[i]:.4f} "
        f"(best achievable per muscle: {scores.max(axis=0)[informative].mean():.4f})"
    )
    return np.full(NUM_DOFS, time_constants[i]), scores[i].mean(axis=1)


def build_loss(
    cfg: omegaconf.DictConfig, train_data: Dataset, rollout_len: int, batch_size: int
) -> PositionAccuracyLoss:
    return PositionAccuracyLoss(
        1,
        train_data,
        train_data,
        None,
        cfg.actuator_net,
        1.0 / cfg.target_frequency,
        batch_size,
        batch_size,
        rollout_len,
        incomplete_histories=True,
        # The built-in normalization rolls the simulation out once per trajectory in Python,
        # which is prohibitive for the long histories this model uses.
        normalize_loss=False,
        stop_gradients_an_input=True,
        end_effector=cfg.end_effector,
    )


@nnx.jit(static_argnames=("loss",))
def sysid_step(
    optimizer: nnx.Optimizer,
    loss: PositionAccuracyLoss,
    permutation: jnp.ndarray,
    batch_idx: int,
) -> jnp.ndarray:
    (batch_loss, _), grads = nnx.value_and_grad(loss, has_aux=True)(
        optimizer.model, permutation, batch_idx
    )
    optimizer.update(grads)
    return batch_loss


def evaluate_model(
    model: AnalyticMuscleEnsemble,
    evaluator: SimulationAccuracyEvaluator,
    summary_writer: SummaryWriter,
    sim_timestep: float,
    step: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (train_errs, _, _, _), (val_errs, val_errs_quant, val_errs_filt, _), _ = (
        evaluator.evaluate(model)
    )
    log_sim_errors(
        summary_writer,
        SIM_EVAL_STEPS,
        [
            ("train/position", train_errs, ""),
            ("val/position", val_errs, ""),
            ("val/position", val_errs_quant, "_quant"),
            ("val/position", val_errs_filt, "_filt"),
        ],
        sim_timestep,
        step,
    )
    return train_errs, val_errs


def select_initialization(
    cfg: omegaconf.DictConfig,
    model_kwargs: Dict[str, Any],
    evaluator: SimulationAccuracyEvaluator,
    seed: int,
) -> AnalyticMuscleEnsemble:
    """Pick the best of several perturbed initializations on the full-horizon rollout error.

    Fitting a rollout error has no cheap warm-up stage and is exposed to local minima, and the
    tendon slack and the membrane elasticity are the two quantities the data sheet constrains
    least, so they are the ones that get resampled. Screening on the evaluator rather than on a
    short training batch matters: a short horizon ranks candidates by local accuracy, which is
    not what limits this model.
    """
    rng = np.random.default_rng(seed)
    best_model = None
    best_error = np.inf
    for restart in range(cfg.num_restarts):
        kwargs = dict(model_kwargs)
        if restart > 0:
            kwargs["init_tendon_slack"] = float(
                model_kwargs["init_tendon_slack"] * np.exp(rng.normal(0.0, 0.4))
            )
            kwargs["elastic_stiffness_scale"] = float(np.exp(rng.normal(0.0, 0.4)))
        candidate = hydra.utils.instantiate(
            cfg.actuator_net, rngs=nnx.Rngs(params=seed), **kwargs
        )
        val_errs = evaluator.evaluate(candidate)[1][0]
        error = float(np.abs(np.asarray(val_errs)).mean())
        logger.info(
            f"Restart {restart}: tendon slack "
            f"{kwargs['init_tendon_slack'] * 1e3:.0f} mm, elastic scale "
            f"{kwargs.get('elastic_stiffness_scale', 1.0):.3f} -> "
            f"mean validation error {error * 180.0 / np.pi:.4f} deg"
        )
        if np.isfinite(error) and error < best_error:
            best_error = error
            best_model = candidate
    assert best_model is not None, "All initializations produced a non-finite error."
    logger.info(
        f"Selected initialization with mean validation error "
        f"{best_error * 180.0 / np.pi:.4f} deg"
    )
    return best_model


def run_sysid(cfg: omegaconf.DictConfig) -> Dict[str, Any]:
    assert not (Path.cwd() / "model").exists(), f"Checkpoint path {Path.cwd() / 'model'} already exists."
    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 2**31 - 1)
    np.random.seed(seed)
    rngs = nnx.Rngs(params=seed + 1, backend=seed + 2, train=seed + 3)
    dt = 1.0 / cfg.target_frequency

    repo_path = Path(__file__).parents[2]
    exp_path = repo_path / "exps" / cfg.exp_name
    check_dataset_consistency(cfg.actuator_net, exp_path)

    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)
    raw_data = raw_data[cfg.num_warmup_trajectories :]
    if cfg.max_num_trajectories is not None:
        raw_data = raw_data[: cfg.max_num_trajectories]

    model_kwargs: Dict[str, Any] = {
        "init_tendon_slack": cfg.actuator_net.init_tendon_slack
    }
    if cfg.calibrate_pressure_lag:
        lag_data = raw_data[: cfg.lag_calibration_trajectories]
        recording_timestep = float(
            np.mean((lag_data.timestamps[:, 1:] - lag_data.timestamps[:, :-1]) % 60.0)
        )
        time_constants, explained = calibrate_pressure_lag(
            lag_data.desired_pressures,
            lag_data.observed_pressures,
            recording_timestep,
            np.asarray(cfg.lag_time_constant_grid),
            shared=cfg.shared_pressure_lag,
        )
        logger.info(
            f"Calibrated pressure lag: time constants {np.round(time_constants * 1e3)} ms, "
            f"explained variance {np.round(explained, 3)}"
        )
        if np.any(time_constants > cfg.max_lag_time_constant):
            logger.warning(
                f"Calibrated time constants {np.round(time_constants * 1e3)} ms exceed the "
                f"{cfg.max_lag_time_constant * 1e3:.0f} ms the "
                f"{cfg.actuator_net.num_history_states * cfg.actuator_net.step_size * 2:.0f} ms "
                f"control history can realise; clipping so the reported value is the one the "
                f"model actually applies."
            )
            time_constants = np.minimum(time_constants, cfg.max_lag_time_constant)
        model_kwargs["init_lag_time_constant"] = time_constants.tolist()

    positions, positions_raw, observed_pressures, controls, _ = process_data(
        raw_data,
        cfg.end_effector,
        cfg.filter,
        cfg.target_frequency,
        cfg.actuation_delay,
        plot_visualizations=False,
        compute_torque_labels=False,
    )
    check_pull_signs(positions, controls, cfg.actuator_net.antagonist_pull_signs)
    dataset = Dataset(positions, positions_raw, observed_pressures, controls, None)

    num_train = int(0.8 * dataset.num_trajectories)
    # Fixed seed so that the split matches the one used to train the learned networks.
    mix_indices = np.random.default_rng(seed=42).permutation(dataset.num_trajectories)
    train_data = dataset[mix_indices[:num_train]]
    val_data = dataset[mix_indices[num_train:]]

    summary_writer = SummaryWriter(str(Path.cwd()))
    evaluator = SimulationAccuracyEvaluator(
        rngs,
        train_data,
        val_data,
        None,
        dt,
        cfg.actuator_net,
        cfg.end_effector,
        num_train_episodes=cfg.sim_acc_eval_episodes,
        num_val_episodes=cfg.sim_acc_eval_episodes,
        num_steps_per_episode=cfg.sim_acc_eval_steps_per_episode,
    )

    model = select_initialization(cfg, model_kwargs, evaluator, seed)

    def record_evaluation(step: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nonlocal best_val_error, best_state
        train_errs, val_errs = evaluate_model(
            model, evaluator, summary_writer, dt, step
        )
        errors = np.abs(np.asarray(val_errs))
        val_error = float(errors.mean())
        # The mean over the rollout hides where the error sits: the first step shows the
        # instantaneous torque accuracy, the last shows the accumulated drift.
        first_step = float(errors[:, 0].mean()) * 180.0 / np.pi
        last_index = min(FULL_ROLLOUT_STEPS, errors.shape[1]) - 1
        last_step = float(errors[:, last_index].mean()) * 180.0 / np.pi
        summary_writer.add_scalar(
            "sim_error_deg_mean/val", val_error * 180.0 / np.pi, step
        )
        is_best = val_error < best_val_error
        if is_best:
            best_val_error = val_error
            _, state = nnx.split(model)
            best_state = jax.tree.map(jnp.copy, state)
        logger.info(
            f"Step {step}: validation error {first_step:.4f} deg after 1 step, "
            f"{last_step:.3f} deg after {last_index + 1} steps "
            f"({(last_index + 1) * dt:.2f} s), mean {val_error * 180.0 / np.pi:.3f} deg"
            + ("  <- new best" if is_best else "")
        )
        return train_errs, val_errs

    best_val_error = np.inf
    best_state = None
    logger.info("Initial simulation accuracy:")
    record_evaluation(0)

    loss = build_loss(cfg, train_data, cfg.rollout_len, cfg.batch_size)
    # A learning rate large enough to make progress early eventually destabilizes the rollout,
    # so the schedule decays it; alpha keeps a small floor rather than stopping the fit dead.
    if cfg.get("lr_decay", False):
        learning_rate = optax.cosine_decay_schedule(
            cfg.learning_rate, cfg.num_steps, alpha=0.05
        )
    else:
        learning_rate = cfg.learning_rate
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(cfg.gradient_clip_norm),
            optax.adam(learning_rate),
        ),
    )
    permutation = loss.get_permutation(rngs)
    logger.info(
        f"Optimizing: rollout_len {cfg.rollout_len}, batch size {cfg.batch_size}, "
        f"lr {cfg.learning_rate}, {cfg.num_steps} steps "
        f"({loss.num_batches} batches available)"
    )
    progress = tqdm(range(1, cfg.num_steps + 1), desc="sysid")
    for step in progress:
        batch_loss = float(
            sysid_step(optimizer, loss, permutation, (step - 1) % loss.num_batches)
        )
        if not np.isfinite(batch_loss):
            # Long rollouts occasionally diverge. The best model so far is already saved, so
            # stopping here keeps it rather than losing the run.
            logger.warning(
                f"Loss became {batch_loss} at step {step}; stopping early and keeping the "
                f"best model so far."
            )
            break
        summary_writer.add_scalar("loss/train", batch_loss, step)
        progress.set_postfix(loss=f"{batch_loss:.4g}")
        if step % cfg.eval_interval == 0:
            record_evaluation(step)

    step = cfg.num_steps
    record_evaluation(step)
    # The rollout objective is non-convex, so keep the best model rather than the last one.
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, best_state)
    train_errs, val_errs = evaluate_model(
        model, evaluator, summary_writer, dt, step + 1
    )
    parameters = model.physical_parameters()
    logger.info(f"Identified parameters:\n{json.dumps(parameters, indent=2)}")
    with Path("physical_parameters.json").open("w") as f:
        json.dump(parameters, f, indent=2)

    if cfg.save_models:
        save_actuator_net_checkpoint(model, Path.cwd() / "model")

    results = {
        "train_pos_errs": np.abs(np.asarray(train_errs)).mean(axis=(0, 2)),
        "val_pos_errs": np.abs(np.asarray(val_errs)).mean(axis=(0, 2)),
    }
    np.savez("results.npz", **results)
    for name, errors in results.items():
        logger.info(
            f"{name}: {errors[0] * 180.0 / np.pi:.4f} deg after 1 step, "
            f"{errors[-1] * 180.0 / np.pi:.3f} deg after {len(errors)} steps "
            f"({len(errors) * dt:.2f} s), mean {errors.mean() * 180.0 / np.pi:.3f} deg"
        )
    summary_writer.close()
    return results


@hydra.main(version_base=None, config_path="config", config_name="sysid_muscle_model")
def main(cfg: omegaconf.DictConfig) -> None:
    run_sysid(cfg)


if __name__ == "__main__":
    main()
