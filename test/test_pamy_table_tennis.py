import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
import scipy

from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend, forward_mjx
from sim2real.rl_tasks.physics_backend.physics_backend import ObjData
from sim2real.util.table_tennis import ball_racket_contact_model

ACTUATOR_NET_EXP_NAME = "2026-05-05_black_heavy_racket"
ACTUATOR_NET_NAME = "msloss1_hist3"


def test_ball_racket_contact_model():
    obj_data = {
        "ball": ObjData(
            jnp.array([[0.0, 0.0, 0.0]]),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
            jnp.array([[-2.0, 1.5, 1.0]]),
            jnp.array([[0.0, 0.0, 0.0]]),
            jnp.array(0.0),
        ),
        "racket": ObjData(
            jnp.array([[0.0, 0.0, -0.03]]),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
            jnp.array([[0.0, 0.0, -1.0]]),
            jnp.array([[0.0, 0.0, 0.0]]),
            jnp.array(0.0),
        ),
    }

    ball_pos, ball_vel = ball_racket_contact_model(obj_data)
    pass


def test_obj_vel():
    rng = jax.random.PRNGKey(42)
    backend = MjxBackend("table_tennis", ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME)
    _, _, state, _ = backend.reset(rng, jnp.zeros((1, 4)), None, None)
    mjx_data = state.mjx_data.replace(qvel=state.mjx_data.qvel.at[:, :4].set(0.0))
    mjx_data = forward_mjx(backend.model, mjx_data)
    obj_data = backend._get_obj_poses_velocities(mjx_data)
    assert obj_data["racket"].linvel == pytest.approx(0.0)
    assert obj_data["racket"].rotvel == pytest.approx(0.0)
    mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[:, 3].set(1.0))
    mjx_data = forward_mjx(backend.model, mjx_data)
    obj_data = backend._get_obj_poses_velocities(mjx_data)
    rotation_to_racket = scipy.spatial.transform.Rotation.from_quat(
        obj_data["racket"].quat
    )
    assert obj_data["racket"].linvel == pytest.approx(0.0)
    assert obj_data["racket"].rotvel == pytest.approx(
        rotation_to_racket.apply(np.array([0.0, 1.0, 0.0])), abs=1e-7
    )
    mjx_data = mjx_data.replace(
        qvel=mjx_data.qvel.at[:, :4].set(jnp.array([0.0, 0.0, 1.0, 0.0]))
    )
    mjx_data = forward_mjx(backend.model, mjx_data)
    obj_data = backend._get_obj_poses_velocities(mjx_data)
    assert jnp.any(obj_data["racket"].linvel > 1e-3)

    ball_linvel = jnp.array([[1.0, 2.0, 3.0]])
    ball_rotvel = jnp.array([[4.0, 5.0, 6.0]])
    mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[:, 4:7].set(ball_linvel))
    mjx_data = mjx_data.replace(qvel=mjx_data.qvel.at[:, 7:10].set(ball_rotvel))
    mjx_data = forward_mjx(backend.model, mjx_data)
    obj_data = backend._get_obj_poses_velocities(mjx_data)
    assert obj_data["ball"].linvel == pytest.approx(ball_linvel)
    assert obj_data["ball"].rotvel == pytest.approx(ball_rotvel)
