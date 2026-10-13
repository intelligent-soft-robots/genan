import jax.numpy as jnp
import numpy as np
import pytest

from sim2real.data_collection.cocontraction import (
    Cocontraction,
)

controls_test_cocontraction = np.array(
    [[-1.0, -0.05, 0.99, 1.0], [-0.5, 0.5, 0.75, 0.8]]
)


@pytest.mark.parametrize(
    "controls,min_pressures_ago,max_pressures_ago,min_pressures_antago,max_pressures_antago",
    [
        (
            controls_test_cocontraction,
            [10000, 11000, 12000, 13000],
            [17000, 16000, 15000, 14000],
            [11000, 10000, 9000, 8000],
            [14000, 15000, 16000, 17000],
        ),
        (
            controls_test_cocontraction,
            [13000, 16400, 12400, 12000],
            [23000, 17500, 23500, 19000],
            [13000, 19000, 12200, 11800],
            [23000, 24500, 22500, 19000],
        ),
        (
            jnp.array(controls_test_cocontraction),
            [13000, 16400, 12400, 12000],
            [23000, 17500, 23500, 19000],
            [13000, 19000, 12200, 11800],
            [23000, 24500, 22500, 19000],
        ),
    ],
)
def test_cocontraction(
    controls,
    min_pressures_ago,
    max_pressures_ago,
    min_pressures_antago,
    max_pressures_antago,
):

    ndp = [0.5, 0.3, 0.4, 0.3]
    cocontraction = Cocontraction(
        min_pressures_ago,
        max_pressures_ago,
        min_pressures_antago,
        max_pressures_antago,
        ndp,
    )

    pressures_ago, pressures_antago = cocontraction.control_to_pressures(controls)
    controls_calc = cocontraction.pressures_to_control(pressures_ago, pressures_antago)

    assert np.asarray(controls_calc) == pytest.approx(np.asarray(controls), abs=1e-3)


@pytest.mark.parametrize(
    "controls",
    [np.array([-1.0, -0.05, 0.05, 1.0]), jnp.array([-1.0, -0.05, 0.05, 1.0])],
)
def test_select_closest_control(controls):
    min_pressures_ago = [10000, 11000, 12000, 13000]
    max_pressures_ago = [17000, 16000, 15000, 14000]
    min_pressures_antago = [11000, 10000, 9000, 8000]
    max_pressures_antago = [14000, 15000, 16000, 17000]
    ndp = [0.9, 0.85, 0.8, 0.75]
    cocontraction = Cocontraction(
        min_pressures_ago,
        max_pressures_ago,
        min_pressures_antago,
        max_pressures_antago,
        ndp,
    )

    pressures_ago, pressures_antago = cocontraction.control_to_pressures(controls)
    controls_calc_unperturbed = cocontraction.pressures_to_control(
        pressures_ago, pressures_antago
    )

    pressures_ago -= np.array([2000, 0, 1100, 0])
    pressures_antago -= np.array([0, 0, 0, 500])

    controls_perturbed = cocontraction.pressures_to_control(
        pressures_ago, pressures_antago
    )

    assert controls_perturbed == pytest.approx(controls_calc_unperturbed, abs=1e-6)
