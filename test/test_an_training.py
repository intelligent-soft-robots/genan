import json
import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import gymnasium
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import optax
import pytest
import skrl
from flax import nnx

from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.loss import (
    PositionAccuracyLoss,
    TorquePredictionLoss,
    get_histories,
)
from sim2real.actuator_network.models import (
    NUM_DOFS,
    ActuatorEnsemble,
    PerDoFActuatorEnsemble,
    restore_actuator_net_checkpoint,
    restore_actuator_net_checkpoint_from_path,
    restore_actuator_net_rl_checkpoint,
    save_actuator_net_checkpoint,
)
from sim2real.actuator_network.pack_an_input import pack_an_input, unpack_an_input
from sim2real.actuator_network.simulation_accuracy_evaluator import (
    SimulationAccuracyEvaluator,
)
from sim2real.actuator_network.standardize import de_standardize, standardize
from sim2real.data_collection.cocontraction import Cocontraction
from sim2real.data_collection.trajectory_dataset import TrajectoryDataset
from sim2real.data_processing.process_data import (
    build_dataset,
    pad_history_length,
    process_data,
)
from sim2real.rl_tasks.agent import load_agent
from sim2real.scripts.train_actuator_network import split_datasets
from sim2real.util.config_utils import (
    get_an_input_size_from_cfg,
    get_an_training_config,
)

skrl.config.jax.backend = "jax"


ACTUATOR_NET_EXP_NAME = "2026-05-05_black_heavy_racket"
ACTUATOR_NET_NAME = "msloss1_hist3"


class IdealActuatorNet(nnx.Module):
    def __init__(self, input_data: jnp.ndarray, output_data: jnp.ndarray):
        self.input_data = jnp.array(input_data)
        self.output_data = jnp.array(output_data)

        self.input_mean = jnp.zeros(self.input_data.shape[-1])
        self.input_std = jnp.ones(self.input_data.shape[-1])
        self.output_mean = jnp.zeros(self.output_data.shape[-1])
        self.output_std = jnp.ones(self.output_data.shape[-1])

    def eval(self):
        return

    def __call__(self, x: jnp.ndarray, rng=None):
        return self.forward_all(x[0])

    def forward_all(
        self, an_input: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, None]:
        indices = jnp.array(
            [
                jnp.argmin(jnp.linalg.norm(inp - self.input_data, axis=1))
                for inp in an_input
            ]
        )
        torques = self.output_data[indices][None]
        return torques, torques, None

    def forward_all_no_jit(self, an_input: jnp.ndarray, rng=None):
        return self.forward_all(an_input)

    @property
    def ensemble_size(self):
        return 1


def do_sgd_step_an_ensemble(rngs: nnx.Rngs, ensemble: ActuatorEnsemble) -> None:
    datapoint_size = ensemble.input_mean.shape[-1]
    optimizer = nnx.Optimizer(ensemble, optax.adamw(3e-4))
    training_in = jax.random.normal(
        rngs.test(), (ensemble.ensemble_size, 42, datapoint_size)
    )
    training_out = jax.random.normal(rngs.test(), (ensemble.ensemble_size, 42, 4))

    def loss(model):
        return jnp.linalg.norm(
            (model(training_in, jax.random.PRNGKey(0))[1] - training_out)
        )

    grads = nnx.grad(loss)(optimizer.model)
    optimizer.update(grads)


def test_standardize():
    ensemble_size = 3
    dataset_size = 10000
    datapoint_size = 5

    scales = np.random.uniform(-42, 42, size=datapoint_size)
    offsets = np.random.uniform(-42, 42, size=datapoint_size)

    key = jax.random.PRNGKey(42)
    key_ne, key_e = jax.random.split(key, 2)
    data_jax_no_ensemble_dim = (
        jax.random.normal(key_ne, (dataset_size, datapoint_size)) * scales + offsets
    )
    data_jax_ensemble_dim = (
        jax.random.normal(key_e, (dataset_size, datapoint_size)) * scales + offsets
    )[None].repeat(ensemble_size, axis=0)
    data_np_no_ensemble_dim = (
        np.random.randn(dataset_size, datapoint_size) * scales + offsets
    )
    data_np_ensemble_dim = (
        np.random.randn(dataset_size, datapoint_size) * scales + offsets
    )[None].repeat(ensemble_size, axis=0)

    for data in [
        data_jax_no_ensemble_dim,
        data_jax_ensemble_dim,
        data_np_no_ensemble_dim,
        data_np_ensemble_dim,
    ]:
        data_mean = data.mean(axis=-2)
        data_std = data.std(axis=-2)
        if data.ndim == 3:
            data_mean = data_mean[0]
            data_std = data_std[0]
        data_standardized = standardize(data, data_mean, data_std)
        assert type(data_standardized) is type(data)
        assert data_standardized.shape == data.shape
        assert data_standardized.mean(axis=-2) == pytest.approx(0.0, abs=1e-5)
        assert data_standardized.std(axis=-2) == pytest.approx(1.0)
        assert de_standardize(data_standardized, data_mean, data_std) == pytest.approx(
            data, rel=1e-5, abs=1e-5
        )


@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize(
    "model_config",
    [
        {"_target_": "sim2real.actuator_network.models.ActuatorEnsemble"},
        {
            "_target_": "sim2real.actuator_network.models.PerDoFActuatorEnsemble",
            "all_dofs_input": False,
        },
        {
            "_target_": "sim2real.actuator_network.models.PerDoFActuatorEnsemble",
            "all_dofs_input": True,
        },
    ],
)
def test_an_save_load(use_batch_norm, model_config):
    an_train_config = {
        **model_config,
        "ensemble_size": 2,
        "num_history_states": 3,
        "step_size": 3,
        "pressure_input": False,
        "velocity_input": True,
        "position_history": False,
        "velocity_history": True,
        "control_history": False,
        "delta_history": True,
        "use_batch_norm": use_batch_norm,
    }
    rngs = nnx.Rngs(test=13, reset=23, backend=42, params=1337)
    input_size = 24
    output_size = 4
    an = hydra.utils.instantiate(
        an_train_config,
        rngs=rngs,
        input_mean=jax.random.uniform(rngs.test(), (input_size,), dtype=jnp.float32),
        input_std=jax.random.uniform(rngs.test(), (input_size,), dtype=jnp.float32),
        output_mean=jax.random.uniform(rngs.test(), (output_size,), dtype=jnp.float32),
        output_std=jax.random.uniform(rngs.test(), (output_size,), dtype=jnp.float32),
    )
    an.train()

    do_sgd_step_an_ensemble(rngs, an)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        save_actuator_net_checkpoint(an, Path(tmpdir) / "model")
        an_loaded = restore_actuator_net_checkpoint_from_path(
            Path(tmpdir) / "model", an_train_config
        )

    assert jnp.all(an.input_mean.value == an_loaded.input_mean.value)
    assert jnp.all(an.input_std.value == an_loaded.input_std.value)
    assert jnp.all(an.output_mean.value == an_loaded.output_mean.value)
    assert jnp.all(an.output_std.value == an_loaded.output_std.value)

    if isinstance(an_loaded, PerDoFActuatorEnsemble):
        input_size_per_dof = (
            input_size if an_train_config["all_dofs_input"] else input_size // NUM_DOFS
        )
        assert an_loaded.layers[0].kernel.shape[:3] == (
            an_train_config["ensemble_size"],
            NUM_DOFS,
            input_size_per_dof,
        )
        assert an_loaded.layers[-1].kernel.shape[-1] == output_size // NUM_DOFS

    data = jax.random.normal(
        rngs.test(), (an_train_config["ensemble_size"], 1000, input_size)
    )
    an.eval()
    an_loaded.eval()
    rng = jax.random.PRNGKey(0)
    assert an(data, rng)[0] == pytest.approx(an_loaded(data, rng)[0], abs=1e-6)


def test_ensemble_eval_mode():
    ensemble_size = 2
    step_size = 3
    num_history_states = 3
    datapoint_size = 4 * (1 + num_history_states) * 3
    rngs = nnx.Rngs(default=42, params=23)
    data = jax.random.normal(rngs(), (1000, datapoint_size))
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    outp_mean = (jax.random.uniform(rngs(), (4,), dtype=jnp.float32) - 0.5) * 84.0
    outp_std = jax.random.uniform(rngs(), (4,), dtype=jnp.float32) * 42.0
    ensemble = ActuatorEnsemble(
        rngs,
        ensemble_size,
        step_size,
        num_history_states,
        False,
        True,
        True,
        True,
        True,
        False,
        activation_fn="elu",
        use_batch_norm=True,
        input_mean=data_mean,
        input_std=data_std,
        output_mean=outp_mean,
        output_std=outp_std,
    )

    ensemble.eval()
    test_sample = jax.random.normal(rngs(), (2, 1, datapoint_size))
    inp1 = jnp.concatenate(
        (test_sample, jax.random.normal(rngs(), (2, 41, datapoint_size))), axis=1
    )
    inp2 = jnp.concatenate(
        (
            test_sample,
            jax.random.normal(rngs(), (2, 41, datapoint_size)) * 0.5 + 2.0,
        ),
        axis=1,
    )

    rng = jax.random.PRNGKey(0)
    out1, _, _ = ensemble(inp1, rng)
    out2, _, _ = ensemble(inp2, rng)

    # In train mode the other samples in the batch might influence the result for the test_sample due to batch
    # normalization, in eval mode, the other samples should not have an influence
    assert out1[:, 0] == pytest.approx(out2[:, 0])


def test_sim_accuracy_evaluator_gt():
    jax.config.update("jax_enable_x64", True)

    exp_path = Path(__file__).parents[1] / "exps" / ACTUATOR_NET_EXP_NAME
    an_train_cfg = get_an_training_config(ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME)
    dt = 1.0 / an_train_cfg.target_frequency

    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)[:400]

    positions, positions_raw, observed_pressures, controls, torques = process_data(
        raw_data,
        an_train_cfg.end_effector,
        an_train_cfg.filter,
        an_train_cfg.target_frequency,
        0,
        plot_visualizations=False,
        precision="double",
    )

    dataset = Dataset(positions, positions_raw, observed_pressures, controls, torques)

    train_data, val_data, _ = split_datasets(an_train_cfg, dataset, dt)

    input_data, output_torques = build_dataset(
        dataset.positions,
        dataset.observed_pressures,
        dataset.controls,
        dataset.torques,
        an_train_cfg.actuator_net,
        dt,
    )
    assert input_data.shape[0] == output_torques.shape[0]

    rngs = nnx.Rngs(backend=42)
    num_train_eps = 200
    num_val_eps = 200
    sim_accuracy_evaluator = SimulationAccuracyEvaluator(
        rngs,
        train_data,
        val_data,
        None,
        dt,
        an_train_cfg.actuator_net,
        an_train_cfg.end_effector,
        num_train_episodes=num_train_eps,
        num_val_episodes=num_val_eps,
        num_steps_per_episode=994,
    )
    (train_err_pos, t1, t2, _), (val_err_pos, _, _, _), _ = (
        sim_accuracy_evaluator.evaluate(IdealActuatorNet(input_data, output_torques))
    )
    mean_abs_errors_train_pos = jnp.abs(train_err_pos).mean(axis=(0, 2))
    mean_abs_errors_val_pos = jnp.abs(val_err_pos).mean(axis=(0, 2))

    assert np.array(mean_abs_errors_train_pos[:200]) == pytest.approx(0.0, abs=1e-6)
    assert np.array(mean_abs_errors_val_pos[:200]) == pytest.approx(0.0, abs=1e-6)

    assert np.array(mean_abs_errors_train_pos[:500]) == pytest.approx(0.0, abs=1e-4)
    assert np.array(mean_abs_errors_val_pos[:500]) == pytest.approx(0.0, abs=1e-4)

    assert np.array(mean_abs_errors_train_pos) == pytest.approx(0.0, abs=1e-1)
    assert np.array(mean_abs_errors_val_pos) == pytest.approx(0.0, abs=1e-1)

    jax.config.update("jax_enable_x64", False)


def test_get_histories():
    num_trajectories = 50
    traj_len = 100
    positions = (
        np.arange(0, num_trajectories)[:, None]
        + np.arange(0, traj_len * 0.001, 0.001)[None]
    )
    positions = np.stack(
        [positions, positions + 0.0001, positions + 0.0002, positions + 0.0003], axis=2
    )
    controls = positions + 0.00001
    dataset = Dataset(positions, positions, None, controls, None)
    rollout_len = 30
    history_length = 10
    indices = np.indices(
        (
            num_trajectories,
            traj_len - history_length - rollout_len,
        )
    )
    histories = get_histories(
        dataset, indices[0], indices[1], history_length, rollout_len
    )
    assert np.all(histories.positions[:, :, :-1] + 0.00001 == histories.controls)
    assert histories.positions[1:, :, :] - histories.positions[
        :-1, :, :
    ] == pytest.approx(1.0)
    assert histories.controls[1:, :, :] - histories.controls[
        :-1, :, :
    ] == pytest.approx(1.0)
    assert np.all(histories.positions[:, 1:, :-1] == histories.positions[:, :-1, 1:])
    assert np.all(histories.controls[:, 1:, :-1] == histories.controls[:, :-1, 1:])
    assert histories.positions[:, :, 1:] - histories.positions[
        :, :, :-1
    ] == pytest.approx(0.001)
    assert histories.controls[:, :, 1:] - histories.controls[
        :, :, :-1
    ] == pytest.approx(0.001)


def test_pos_acc_loss_gt():
    exp_path = Path(__file__).parents[1] / "exps" / ACTUATOR_NET_EXP_NAME
    an_train_cfg = get_an_training_config(ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME)
    dt = 1.0 / an_train_cfg.target_frequency

    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)

    raw_data = raw_data[:20]

    filter_cfg = {"_target_": "sim2real.data_processing.filter.filter.DefaultFilter"}
    positions, positions_raw, observed_pressures, controls, torques = process_data(
        raw_data,
        an_train_cfg.end_effector,
        filter_cfg,
        an_train_cfg.target_frequency,
        an_train_cfg.get("actuation_delay", 0),
    )

    dataset = Dataset(
        positions,
        positions_raw,
        observed_pressures,
        controls,
        torques,
    )
    train_data, val_data, _ = split_datasets(an_train_cfg, dataset, dt)
    input_data, output_torques = build_dataset(
        positions, observed_pressures, controls, torques, an_train_cfg.actuator_net, dt
    )

    batch_size = 512
    loss = PositionAccuracyLoss(
        1,
        train_data,
        val_data,
        val_data,
        an_train_cfg.actuator_net,
        dt,
        batch_size,
        batch_size,
        30,
        incomplete_histories=False,
        normalize_loss=False,
        stop_gradients_an_input=True,
        end_effector=an_train_cfg.end_effector,
    )
    ideal_an = IdealActuatorNet(input_data, output_torques)
    permutation = loss.get_permutation(nnx.Rngs(42))
    train_loss = loss(ideal_an, permutation, 0)[0]
    val_loss = loss.val_loss(ideal_an)
    extra_eval_loss = loss.extra_eval_loss(ideal_an)

    assert train_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert val_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert extra_eval_loss.item() == pytest.approx(0.0, abs=1e-5)


def test_torque_pred_loss_gt():
    exp_path = Path(__file__).parents[1] / "exps" / ACTUATOR_NET_EXP_NAME
    an_train_cfg = get_an_training_config(ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME)
    dt = 1.0 / an_train_cfg.target_frequency

    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)

    raw_data = raw_data[:20]

    filter_cfg = {"_target_": "sim2real.data_processing.filter.filter.DefaultFilter"}
    positions, positions_raw, observed_pressures, controls, torques = process_data(
        raw_data,
        an_train_cfg.end_effector,
        filter_cfg,
        an_train_cfg.target_frequency,
        an_train_cfg.get("actuation_delay", 0),
    )

    dataset = Dataset(
        positions,
        positions_raw,
        observed_pressures,
        controls,
        torques,
    )
    train_data, val_data, _ = split_datasets(an_train_cfg, dataset, dt)
    input_data, output_torques = build_dataset(
        positions, observed_pressures, controls, torques, an_train_cfg.actuator_net, dt
    )

    batch_size = 512
    loss = TorquePredictionLoss(
        1,
        train_data,
        val_data,
        val_data,
        an_train_cfg.actuator_net,
        dt,
        batch_size,
        batch_size,
        huber_delta=None,
        max_episode_length_s=None,
    )
    ideal_an = IdealActuatorNet(input_data, output_torques)
    permutation = loss.get_permutation(nnx.Rngs(42))
    train_loss = loss(ideal_an, permutation, 0)[0]
    val_loss = loss.val_loss(ideal_an)
    extra_eval_loss = loss.extra_eval_loss(ideal_an)

    assert train_loss.item() == 0.0
    assert val_loss.item() == 0.0
    assert extra_eval_loss.item() == 0.0


@pytest.mark.parametrize(
    "delta_history, velocity_input, pressure_input, position_history, velocity_history, control_history",
    [
        (False, False, False, True, True, True),
        (True, False, True, True, True, True),
        (True, True, False, True, True, True),
        (True, True, True, True, True, True),
        # The *_history flags shrink individual blocks to the newest entry only.
        (False, True, False, False, False, True),
        (True, True, False, False, False, True),
        (False, True, False, True, False, False),
        (False, False, False, False, True, False),
    ],
)
def test_pack_unpack(
    delta_history,
    velocity_input,
    pressure_input,
    position_history,
    velocity_history,
    control_history,
):
    dt = 0.001
    positions = np.array(
        [
            [0.0, 0.01, 0.02, 0.04],
            [0.10, 0.11, 0.12, 0.14],
            [0.20, 0.21, 0.22, 0.24],
            [0.30, 0.31, 0.32, 0.34],
        ]
    ).T
    pressures = np.moveaxis(
        np.array(
            [
                [[1.0, 1.001], [1.01, 1.011], [1.02, 1.031]],
                [[1.1, 1.101], [1.11, 1.111], [1.12, 1.131]],
                [[1.2, 1.201], [1.21, 1.211], [1.22, 1.231]],
                [[1.3, 1.301], [1.31, 1.311], [1.32, 1.331]],
            ]
        ),
        0,
        1,
    )
    controls = np.array(
        [[2.0, 2.01, 2.03], [2.10, 2.11, 2.13], [2.20, 2.21, 2.23], [2.30, 2.31, 2.33]]
    ).T
    actuator_net_cfg = {
        "num_history_states": 2,
        "step_size": 1,
        "pressure_input": pressure_input,
        "velocity_input": velocity_input,
        "delta_history": delta_history,
        "position_history": position_history,
        "velocity_history": velocity_history,
        "control_history": control_history,
    }
    test_input = pack_an_input(
        positions,
        pressures,
        controls,
        actuator_net_cfg,
        dt,
    )
    test_output = unpack_an_input(test_input, actuator_net_cfg)

    expected_positions = positions[1:] if position_history else positions[-1:]
    assert test_output["position_context"] == pytest.approx(expected_positions)
    if velocity_input:
        velocities = (positions[1:] - positions[:-1]) / dt
        expected_velocities = velocities if velocity_history else velocities[-1:]
        assert test_output["velocity_context"] == pytest.approx(expected_velocities)
    else:
        assert test_output["velocity_context"] is None
    if pressure_input:
        assert test_output["pressure_context"] == pytest.approx(pressures)
    else:
        assert test_output["pressure_context"] is None
    expected_controls = controls if control_history else controls[-1:]
    assert test_output["control_context"] == pytest.approx(expected_controls)


def test_delta_history():
    dt = 1.0
    positions = np.array(
        [
            [0.0, 0.00, 0.01, 0.03],
            [0.0, 0.10, 0.11, 0.13],
            [0.0, 0.20, 0.21, 0.23],
            [0.0, 0.30, 0.31, 0.33],
        ]
    ).T
    pressures = np.moveaxis(
        np.array(
            [
                [[1.0, 2.0], [1.01, 2.01], [1.03, 2.03]],
                [[1.1, 2.1], [1.11, 2.11], [1.13, 2.13]],
                [[1.2, 2.2], [1.21, 2.21], [1.23, 2.23]],
                [[1.3, 2.3], [1.31, 2.31], [1.33, 2.33]],
            ]
        ),
        0,
        1,
    )
    controls = np.array(
        [[3.0, 3.01, 3.03], [3.10, 3.11, 3.13], [3.20, 3.21, 3.23], [3.30, 3.31, 3.33]]
    ).T
    an_cfg = {
        "num_history_states": 2,
        "step_size": 1,
        "pressure_input": True,
        "velocity_input": False,
        "position_history": True,
        "velocity_history": False,
        "control_history": True,
        "delta_history": True,
    }
    test_input = pack_an_input(
        positions,
        pressures,
        controls,
        an_cfg,
        dt,
    )

    assert test_input[:: an_cfg["num_history_states"] + 1] == pytest.approx(-0.03)
    assert test_input[1 :: an_cfg["num_history_states"] + 1] == pytest.approx(-0.02)
    assert test_input[2 :: an_cfg["num_history_states"] + 1] % 0.1 == pytest.approx(
        0.03
    )


def test_pack_padding_jax_numpy():
    dt = 0.001
    key = jax.random.PRNGKey(42)
    key_pos, key_vel, key_press, key_control = jax.random.split(key, 4)
    positions = jax.random.normal(key_pos, (10, 5, 4))
    pressures = jax.random.normal(key_press, (10, 4, 4, 2))
    controls = jax.random.normal(key_control, (10, 4, 4))
    an_cfg = {
        "num_history_states": 3,
        "step_size": 3,
        "pressure_input": True,
        "velocity_input": True,
        "position_history": True,
        "velocity_history": True,
        "control_history": True,
        "delta_history": True,
    }
    an_input_jax = pack_an_input(
        positions,
        pressures,
        controls,
        an_cfg,
        dt,
        pad_incomplete_hist=True,
    )
    an_input_numpy = pack_an_input(
        np.array(positions),
        np.array(pressures),
        np.array(controls),
        an_cfg,
        dt,
        pad_incomplete_hist=True,
    )

    assert an_input_numpy == pytest.approx(np.array(an_input_jax))


def test_sim_acc_eval_loss():
    dt = 0.002
    traj_len = 20

    exp_path = Path(__file__).parents[1] / "exps" / ACTUATOR_NET_EXP_NAME
    actuator_net_path = exp_path / "actuator_network" / ACTUATOR_NET_NAME

    train_actuator_net_cfg = omegaconf.OmegaConf.load(
        actuator_net_path / "train_actuator_network_config" / "config.yaml"
    )
    an_cfg = train_actuator_net_cfg.actuator_net

    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)

    positions, positions_raw, observed_pressures, controls, _ = process_data(
        raw_data,
        train_actuator_net_cfg.end_effector,
        train_actuator_net_cfg.filter,
        train_actuator_net_cfg.target_frequency,
        train_actuator_net_cfg.get("actuation_delay", 0),
    )

    dataset = Dataset(
        positions,
        positions_raw,
        observed_pressures,
        controls,
        np.nan * np.ones_like(positions),
    )

    train_data, val_data, train_data_stats = split_datasets(
        train_actuator_net_cfg, dataset, dt, train_split=1.0
    )
    actuator_ensemble = restore_actuator_net_checkpoint(
        ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME
    )

    sim_acc_eval = SimulationAccuracyEvaluator(
        nnx.Rngs(42),
        train_data,
        val_data,
        None,
        0.002,
        an_cfg,
        train_actuator_net_cfg.end_effector,
        200,
        0,
        num_steps_per_episode=traj_len,
        step_all_models=True,
    )

    dataset_loss = Dataset(
        sim_acc_eval.positions,
        sim_acc_eval.positions_raw,
        np.nan * np.ones((*sim_acc_eval.positions.shape, 2)),
        jnp.concatenate(
            (
                sim_acc_eval.controls,
                jnp.nan
                * jnp.ones(
                    (sim_acc_eval.controls.shape[0], 1, sim_acc_eval.controls.shape[2]),
                    dtype=jnp.float32,
                ),
            ),
            axis=1,
        ),
        np.nan * np.ones_like(sim_acc_eval.positions),
    )

    position_accuracy_loss = PositionAccuracyLoss(
        actuator_ensemble.ensemble_size,
        dataset_loss,
        val_data,  # Unused
        None,
        an_cfg,
        sim_acc_eval.backend.dt,
        200,
        10000,
        traj_len,
        incomplete_histories=False,
        normalize_loss=False,
        stop_gradients_an_input=False,
        end_effector=train_actuator_net_cfg.end_effector,
    )

    # Exactly enough data for one batch --> This batch will be a permutation of the data used by sim_acc_eval
    permutation = position_accuracy_loss.get_permutation(nnx.Rngs(42))

    actuator_ensemble.eval()
    loss, loss_models = position_accuracy_loss(actuator_ensemble, permutation, 0)
    loss = np.array(loss)
    loss_models = np.array(loss_models)

    (pos_errs, _, _, _), _, _ = sim_acc_eval.evaluate(actuator_ensemble)
    loss_eval = np.array((pos_errs[:, :, :traj_len] ** 2).mean(axis=(0, 2, 3)))

    assert loss_models == pytest.approx(loss_eval)
    assert loss == pytest.approx(np.mean(loss_eval))


def test_restore_checkpoint_rl():
    actuator_net_name = "rl30_step1_hist3"
    actuator_net = restore_actuator_net_rl_checkpoint(
        ACTUATOR_NET_EXP_NAME, actuator_net_name
    )
    actuator_net_train_cfg = get_an_training_config(
        ACTUATOR_NET_EXP_NAME, actuator_net_name, rl_training=True
    )
    train_cfg = omegaconf.OmegaConf.create(
        omegaconf.OmegaConf.to_container(actuator_net_train_cfg, resolve=True)
    )

    inp_size = get_an_input_size_from_cfg(train_cfg.actuator_net)
    dummy_observation_space = gymnasium.spaces.Box(
        -np.inf * np.ones(inp_size, dtype=np.float32),
        np.inf * np.ones(inp_size, dtype=np.float32),
    )
    dummy_action_space = gymnasium.spaces.Box(
        -np.inf * np.ones(4, dtype=np.float32),
        np.inf * np.ones(4, dtype=np.float32),
    )

    exps_path = Path(__file__).parents[1] / "exps"
    checkpoint_path = (
        exps_path
        / ACTUATOR_NET_EXP_NAME
        / "actuator_network_rl"
        / actuator_net_name
        / "checkpoints"
        / "best_agent.pickle"
    )
    agent = load_agent(
        checkpoint_path,
        train_cfg.agent,
        dummy_observation_space,
        dummy_action_space,
        1,
    )

    dummy_input = jax.random.normal(
        jax.random.PRNGKey(42), (1, inp_size), dtype=jnp.float32
    )
    output_rl = agent.policy.apply(
        agent.policy.state_dict.params,
        {"states": agent._state_preprocessor(dummy_input)},
        "policy",
    )[0]
    output_an = actuator_net(dummy_input)[0]

    assert jnp.all(output_rl == output_an)


@pytest.mark.parametrize(
    "num_history_states, step_size, delta_history",
    [(0, 1, True), (3, 1, True), (3, 1, False), (3, 3, True), (3, 3, False)],
)
def test_pad_dataset(
    num_history_states: int, step_size: int, delta_history: bool
) -> None:
    traj_len = 20
    dt = 0.002
    seq = np.arange(1, traj_len + 2)
    seq_ctrl = np.arange(2, traj_len + 2)
    positions = np.stack([seq + 0.00, seq + 0.01, seq + 0.02, seq + 0.03], axis=1)[None]
    positions_raw = np.stack([seq + 0.10, seq + 0.11, seq + 0.12, seq + 0.13], axis=1)[
        None
    ]
    observed_pressures = np.stack(
        [seq + 0.20, seq + 0.21, seq + 0.22, seq + 0.23], axis=1
    )[None, :, :, None].repeat(2, axis=3)
    controls = np.stack(
        [seq_ctrl + 0.30, seq_ctrl + 0.31, seq_ctrl + 0.32, seq_ctrl + 0.33], axis=1
    )[None]

    an_cfg = {
        "num_history_states": num_history_states,
        "step_size": step_size,
        "pressure_input": False,
        "velocity_input": True,
        "position_history": True,
        "velocity_history": True,
        "control_history": True,
        "delta_history": delta_history,
    }
    hist_len = an_cfg["num_history_states"] * an_cfg["step_size"] + 1

    positions_pad, positions_raw_pad, observed_pressures_pad, controls_pad = (
        pad_history_length(
            positions, positions_raw, observed_pressures, controls, hist_len
        )
    )

    for idx in range(traj_len):
        start_idx = max(idx - hist_len + 1, 0)
        an_input_not_padded = pack_an_input(
            positions[:, start_idx : idx + 2],
            None,
            controls[:, start_idx : idx + 1],
            an_cfg,
            dt,
            pad_incomplete_hist=True,
        )

        an_input_padded = pack_an_input(
            positions_pad[:, idx : idx + hist_len + 1],
            None,
            controls_pad[:, idx : idx + hist_len],
            an_cfg,
            dt,
            pad_incomplete_hist=False,
        )

        assert np.all(an_input_padded == an_input_not_padded)


def test_actuation_delay():
    traj_len = 20
    seq = np.arange(1, traj_len + 2, dtype=np.float32)
    seq_ctrl = np.arange(1, traj_len + 1, dtype=np.float32)
    positions = np.stack([seq + 0.00, seq + 0.01, seq + 0.02, seq + 0.03], axis=1)[None]
    desired_pressures = np.stack(
        [seq_ctrl + 0.10, seq_ctrl + 0.11, seq_ctrl + 0.12, seq_ctrl + 0.13], axis=1
    )[None, :, :, None].repeat(2, axis=3)
    observed_pressures = np.stack(
        [seq + 0.20, seq + 0.21, seq + 0.22, seq + 0.23], axis=1
    )[None, :, :, None].repeat(2, axis=3)
    controls = np.stack(
        [seq_ctrl + 0.30, seq_ctrl + 0.31, seq_ctrl + 0.32, seq_ctrl + 0.33], axis=1
    )[None]
    timestamps = seq[None] * 0.001
    data = TrajectoryDataset.from_data(
        positions, desired_pressures, observed_pressures, controls, timestamps
    )
    filter_cfg = {"_target_": "sim2real.data_processing.filter.filter.DefaultFilter"}
    p_a0, pr_a0, op_a0, c_a0, _ = process_data(data, "racket", filter_cfg, 1000.0, 0)
    assert p_a0 == pytest.approx(positions)
    assert pr_a0 == pytest.approx(positions)
    assert op_a0 == pytest.approx(observed_pressures)
    assert c_a0 == pytest.approx(controls)

    p_a1, pr_a1, op_a1, c_a1, _ = process_data(data, "racket", filter_cfg, 1000.0, 1)
    assert p_a1 == pytest.approx(positions[:, :-1])
    assert pr_a1 == pytest.approx(positions[:, :-1])
    assert op_a1 == pytest.approx(observed_pressures[:, :-1])
    assert c_a1 == pytest.approx(p_a1[:, :-1] + 1.3)

    p_a1_s2, pr_a1_s2, op_a1_s2, c_a1_s2, _ = process_data(
        data, "racket", filter_cfg, 500.0, 1
    )
    assert p_a1_s2 == pytest.approx(positions[:, 1::2])
    assert pr_a1_s2 == pytest.approx(positions[:, 1::2])
    assert op_a1_s2 == pytest.approx(observed_pressures[:, 1::2])
    assert c_a1_s2 == pytest.approx(p_a1_s2[:, :-1] + 1.3)

    p_a2_s2, pr_a2_s2, op_a2_s2, c_a2_s2, _ = process_data(
        data, "racket", filter_cfg, 500.0, 2
    )
    assert p_a2_s2 == pytest.approx(positions[:, :-2:2])
    assert pr_a2_s2 == pytest.approx(positions[:, :-2:2])
    assert op_a2_s2 == pytest.approx(observed_pressures[:, :-2:2])
    assert c_a2_s2 == pytest.approx(p_a2_s2[:, :-1] + 2.3)

    p_a3_s2, pr_a3_s2, op_a3_s2, c_a3_s2, _ = process_data(
        data, "racket", filter_cfg, 500.0, 3
    )
    assert p_a3_s2 == pytest.approx(positions[:, 1:-2:2])
    assert pr_a3_s2 == pytest.approx(positions[:, 1:-2:2])
    assert op_a3_s2 == pytest.approx(observed_pressures[:, 1:-2:2])
    assert c_a3_s2 == pytest.approx(p_a3_s2[:, :-1] + 3.3)


def test_controls_desired_pressures():
    exp_path = Path(__file__).parents[1] / "exps" / ACTUATOR_NET_EXP_NAME
    data_collection_config_path = exp_path / "collect_data_config" / "config.yaml"
    data_collection_cfg = omegaconf.OmegaConf.load(data_collection_config_path)
    robot_cfg_path = exp_path / "robot_config.json"
    with robot_cfg_path.open("r") as f:
        robot_cfg = json.load(f)

    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)

    cocontraction = Cocontraction(
        robot_cfg["min_pressures_ago"],
        robot_cfg["max_pressures_ago"],
        robot_cfg["min_pressures_antago"],
        robot_cfg["max_pressures_antago"],
        data_collection_cfg["ndp"],
    )

    pressures_controls = np.stack(
        cocontraction.control_to_pressures(raw_data.controls), axis=3
    )

    assert np.all(
        np.isclose(
            pressures_controls, raw_data.desired_pressures[:, :-1].astype(int), atol=1
        )
    )


def test_id_repeatability():
    exp_path = Path(__file__).parents[1] / "exps" / ACTUATOR_NET_EXP_NAME
    an_train_cfg = get_an_training_config(ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME)
    with (exp_path / "raw_data.pkl").open("rb") as f:
        raw_data = pickle.load(f)
    raw_data = raw_data[:20]
    filter_cfg = {"_target_": "sim2real.data_processing.filter.filter.DefaultFilter"}
    _, _, _, _, torques1 = process_data(
        raw_data, an_train_cfg.end_effector, filter_cfg, 500.0, 0, False
    )
    _, _, _, _, torques2 = process_data(
        raw_data, an_train_cfg.end_effector, filter_cfg, 500.0, 0, False
    )

    assert torques1[:, 1:-1] == pytest.approx(torques2[:, 1:-1])
