import math
from pathlib import Path

import hydra
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from sim2real.actuator_network.analytic_muscle_model import (
    DMSP20_INNER_DIAMETER,
    DMSP20_MAX_CONTRACTION,
    DMSP20_MAX_PRESSURE,
    DMSP20_THEORETICAL_FORCE,
    AnalyticMuscleEnsemble,
    datasheet_force_curve_parameters,
    muscle_force,
)
from sim2real.actuator_network.models import (
    restore_actuator_net_checkpoint_from_path,
    save_actuator_net_checkpoint,
)
from sim2real.actuator_network.pack_an_input import pack_an_input, unpack_an_input
from sim2real.util.config_utils import get_an_input_size_from_cfg

CONFIG_PATH = "../sim2real/scripts/config/actuator_net"
ACTUATOR_NET_CONFIGS = ["analytic_muscle"]


def load_actuator_net_config(name: str):
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        return hydra.compose(config_name=name)


def make_model(name: str = "analytic_muscle", **overrides) -> AnalyticMuscleEnsemble:
    cfg = load_actuator_net_config(name)
    return hydra.utils.instantiate(cfg, rngs=nnx.Rngs(params=0), **overrides)


def make_an_input(cfg, num_samples: int, positions=None, controls=None) -> jnp.ndarray:
    hist_len = cfg.num_history_states * cfg.step_size + 1
    if positions is None:
        positions = np.random.uniform(-0.5, 0.5, (num_samples, hist_len + 1, 4))
    if controls is None:
        controls = np.random.uniform(-1.0, 1.0, (num_samples, hist_len, 4))
    return pack_an_input(
        jnp.array(positions, dtype=jnp.float32),
        None,
        jnp.array(controls, dtype=jnp.float32),
        cfg,
        1.0 / 500.0,
    )


def evaluate_force(pressure: float, contraction: float) -> float:
    a, b, elastic_stiffness = datasheet_force_curve_parameters()
    area = math.pi * DMSP20_INNER_DIAMETER**2 / 4.0
    return float(
        muscle_force(
            jnp.array(pressure),
            jnp.array(contraction),
            area,
            jnp.array(a),
            jnp.array(b),
            jnp.array(1.0),
            jnp.array(elastic_stiffness),
        )
    )


def test_force_curve_matches_datasheet():
    assert evaluate_force(DMSP20_MAX_PRESSURE, 0.0) == pytest.approx(
        DMSP20_THEORETICAL_FORCE, rel=1e-3
    )
    assert evaluate_force(DMSP20_MAX_PRESSURE, DMSP20_MAX_CONTRACTION) == pytest.approx(
        0.0, abs=1.0
    )


def test_force_curve_monotonicity():
    contractions = np.linspace(0.0, DMSP20_MAX_CONTRACTION, 26)
    forces = [evaluate_force(3e5, h) for h in contractions]
    assert np.all(np.diff(forces) <= 1e-6)

    pressures = np.linspace(0.0, DMSP20_MAX_PRESSURE, 13)
    forces = [evaluate_force(p, 0.1) for p in pressures]
    assert np.all(np.diff(forces) >= -1e-6)


def test_force_is_zero_without_pressure():
    """An unpressurized muscle cannot contract, so its wire is slack at any contraction."""
    for contraction in [0.0, 0.05, 0.1, 0.25]:
        assert evaluate_force(0.0, contraction) == 0.0


def test_slack_at_rest_produces_no_torque():
    """At zero pressure both muscles of every pair hang slack, whatever the joint angle."""
    cfg = load_actuator_net_config("analytic_muscle")
    model = make_model("analytic_muscle")
    hist_len = cfg.num_history_states * cfg.step_size + 1
    positions = np.tile(np.linspace(-1.0, 1.0, 9)[:, None, None], (1, hist_len + 1, 4))
    controls = np.zeros((9, hist_len, 4))
    an_input = make_an_input(cfg, 9, positions, controls)

    contexts = unpack_an_input(an_input, model.actuator_net_cfg)
    zero_pressures = jnp.zeros_like(model.pressures(contexts["control_context"]))
    forces = muscle_force(
        zero_pressures,
        model.contraction_ratios(contexts["position_context"][..., -1, :]),
        model.muscle_area,
        jnp.exp(model.log_a.value),
        jnp.exp(model.log_b.value),
        jnp.exp(model.log_contraction_correction.value),
        jnp.exp(model.log_elastic_stiffness.value),
    )
    assert np.all(np.asarray(forces) == 0.0)


def test_cocontraction_increases_stiffness():
    """Raising both pressures at a fixed angle must raise the restoring torque gradient."""
    cfg = load_actuator_net_config("analytic_muscle")
    hist_len = cfg.num_history_states * cfg.step_size + 1
    angles = np.array([-0.1, 0.1])

    stiffnesses = []
    for ndp in [0.2, 0.6]:
        model = make_model("analytic_muscle", ndp=[ndp] * 4)
        positions = np.tile(angles[:, None, None], (1, hist_len + 1, 4))
        controls = np.zeros((2, hist_len, 4))
        torques = np.asarray(
            model.forward_all_no_jit(make_an_input(cfg, 2, positions, controls))[0]
        )[0]
        stiffnesses.append(-(torques[1] - torques[0]) / (angles[1] - angles[0]))
    assert np.all(np.array(stiffnesses[1]) > np.array(stiffnesses[0]))


@pytest.mark.parametrize("config_name", ACTUATOR_NET_CONFIGS)
def test_actuator_ensemble_interface(config_name):
    cfg = load_actuator_net_config(config_name)
    model = make_model(config_name)
    an_input = make_an_input(cfg, 6)

    assert an_input.shape[-1] == get_an_input_size_from_cfg(cfg)
    assert model.ensemble_size == 1
    model.eval()

    destandardized, standardized, aux = model.forward_all_no_jit(an_input, None)
    assert destandardized.shape == (1, 6, 4)
    assert standardized.shape == (1, 6, 4)
    assert aux is None
    # The model works in physical units, so standardization is the identity.
    assert np.allclose(np.asarray(destandardized), np.asarray(standardized))

    # step_all_models=True passes an input that already carries the ensemble axis.
    from_call = model(an_input[None], None)[0]
    assert from_call.shape == (1, 6, 4)
    assert np.allclose(np.asarray(from_call), np.asarray(destandardized))


@pytest.mark.parametrize("config_name", ACTUATOR_NET_CONFIGS)
def test_checkpoint_round_trip(config_name, tmp_path):
    cfg = load_actuator_net_config(config_name)
    model = make_model(config_name)
    an_input = make_an_input(cfg, 5)

    checkpoint_path = Path(tmp_path) / "model"
    save_actuator_net_checkpoint(model, checkpoint_path)
    restored = restore_actuator_net_checkpoint_from_path(checkpoint_path, cfg)

    assert np.allclose(
        np.asarray(model.forward_all_no_jit(an_input, None)[0]),
        np.asarray(restored.forward_all_no_jit(an_input, None)[0]),
    )


def test_gradients_only_reach_trainable_parameters():
    cfg = load_actuator_net_config("analytic_muscle")
    model = make_model("analytic_muscle")
    an_input = make_an_input(cfg, 4)

    grads = nnx.grad(lambda m: (m.forward_all_no_jit(an_input, None)[0] ** 2).mean())(
        model
    )
    names = {str(path[0]) for path, _ in nnx.to_flat_state(grads)}
    assert "log_tendon_slack" in names
    assert "log_a" in names
    assert "torque_bias" in names
    # Frozen by default, so they must not receive gradients.
    assert "log_moment_arms" not in names


def test_force_gains_are_trainable_when_requested():
    """Moment arms are never fitted: r and g multiply, so only their product is identifiable."""
    cfg = load_actuator_net_config("analytic_muscle")
    an_input = make_an_input(cfg, 4)
    for fit_force_gains, expected in [(False, False), (True, True)]:
        model = make_model("analytic_muscle", fit_force_gains=fit_force_gains)
        grads = nnx.grad(
            lambda m: (m.forward_all_no_jit(an_input, None)[0] ** 2).mean()
        )(model)
        names = {str(path[0]) for path, _ in nnx.to_flat_state(grads)}
        assert ("log_force_gains" in names) is expected
        assert "log_moment_arms" not in names


def test_lag_kernel_is_normalized_and_decaying():
    model = make_model("analytic_muscle")
    kernel = np.asarray(model.lag_kernel())
    assert kernel.shape == (model.num_history_states + 1, 4)
    assert np.allclose(kernel.sum(axis=0), 1.0)
    assert np.all(kernel >= 0.0)
    # A first-order lag weights the newest sample most and decays monotonically into the past.
    assert np.all(np.diff(kernel, axis=0) > 0.0)


def test_lag_kernel_stays_normalized_for_a_lag_longer_than_the_window():
    """Normalizing the raw weights would underflow and silently scale the pressures down."""
    cfg = load_actuator_net_config("analytic_muscle")
    window = cfg.num_history_states * cfg.step_size * 0.002
    model = make_model("analytic_muscle", init_lag_time_constant=50.0 * window)
    kernel = np.asarray(model.lag_kernel())
    assert np.allclose(kernel.sum(axis=0), 1.0)


def test_lag_kernel_shortens_with_the_time_constant():
    lags = np.asarray(make_model("analytic_muscle").context_lags.value)
    delays = [
        float(
            (
                lags
                * np.asarray(
                    make_model(
                        "analytic_muscle", init_lag_time_constant=tau
                    ).lag_kernel()
                )[:, 0]
            ).sum()
        )
        for tau in [0.005, 0.02, 0.06]
    ]
    assert delays[0] < delays[1] < delays[2]
