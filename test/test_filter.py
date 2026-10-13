import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from sim2real.util.encoder_simulation import ENCODER_RESOLUTION, encoder_quantization


def test_encoder_quantization_real():
    raw_data = (
        Path(__file__).parents[1]
        / "exps"
        / "2026-05-05_black_heavy_racket"
        / "raw_data.pkl"
    )
    with raw_data.open("rb") as f:
        data = pickle.load(f)[:100]
    positions_quantized_np = encoder_quantization(data.positions)
    positions_quantized_jnp = encoder_quantization(jnp.array(data.positions))
    assert positions_quantized_np == pytest.approx(data.positions)
    assert np.array(positions_quantized_jnp) == pytest.approx(data.positions)


def test_encoder_quantization_np_jnp():
    positions_np = ((np.random.rand(20, 6, 4) - 0.5) * 2 * np.pi).astype(np.float32)
    positions_jnp = jnp.array(positions_np)
    positions_quantized_np = encoder_quantization(positions_np)
    positions_quantized_jnp = encoder_quantization(positions_jnp)
    assert positions_quantized_np == pytest.approx(positions_quantized_jnp)
