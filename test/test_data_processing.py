import numpy as np
import pytest

from sim2real.data_processing.finite_differences import (
    central_differences_first,
    central_differences_second,
)


def test_finite_differences():
    slope = np.random.uniform(-10, 10, size=(2, 1, 4))
    offset = np.random.uniform(-10, 10, size=(2, 1, 4))
    t = np.array(
        [
            [0.0, 0.002, 0.004, 0.008, 0.009, 0.0095, 0.01],
            [0.1, 0.5, 0.6, 0.65, 0.7, 0.8, 1.0],
        ]
    )
    x = slope * t[:, :, None] + offset
    assert central_differences_first(x, t)[:, 1:-1] == pytest.approx(
        np.tile(slope, (1, 5, 1))
    )
    assert central_differences_second(x, t)[:, 1:-1] == pytest.approx(0.0, abs=1e-8)

    a = np.random.uniform(-10, 10, size=(2, 1, 4))
    b = np.random.uniform(-10, 10, size=(2, 1, 4))
    c = np.random.uniform(-10, 10, size=(2, 1, 4))

    x = a * t[:, :, None] ** 2 + b * t[:, :, None] + c
    # Central differences always calculate the derivative at the midpoint between t_(i-1) and t_(i+1). If the grid is
    # not evenly split, then this is not t_i.
    t_mid = 0.5 * (t[:, 2:] + t[:, :-2])
    x_der_mid = 2 * a * t_mid[:, :, None] + b

    assert central_differences_first(x, t)[:, 1:-1] == pytest.approx(x_der_mid)
    assert central_differences_second(x, t)[:, 1:-1] == pytest.approx(
        np.tile(2 * a, (1, 5, 1)), abs=1e-8
    )


def test_central_differences_non_regular_grid():
    t = np.array([[1.0, 2.0, 4.0]])
    p = np.array([[1.0, 2.0, 3.0]])

    assert central_differences_second(p, t, keep_len=False) == pytest.approx(-1 / 3)
