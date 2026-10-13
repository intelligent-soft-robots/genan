import jax
import jax.numpy as jnp
import math
import numpy as np
import pytest
from flax import nnx

from sim2real.actuator_network.pack_an_input import pack_an_input
from sim2real.rl_tasks.pamy_reacher.pamy_reacher import PamyReacherEnv
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend
from sim2real.util.config_utils import get_an_training_config

ACTUATOR_NET_EXP_NAME = "2026-05-05_black_heavy_racket"
ACTUATOR_NET_NAME = "msloss1_hist3"


def test_observations():
    num_history_states = 3
    delta_history = False
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    env = PamyReacherEnv(
        backend,
        num_instances=1,
        num_history_states=num_history_states,
        delta_history=delta_history,
    )
    num_dofs = 4

    try:
        hist_len = num_history_states + 1
        positions = jax.random.uniform(
            rngs.env(),
            (hist_len, 1, 4),
            minval=env.workspace_limits_lower,
            maxval=env.workspace_limits_upper,
            dtype=jnp.float32,
        )
        velocities = (positions[1:] - positions[:-1]) / env.backend.dt
        actions = jax.random.uniform(
            rngs.env(),
            (hist_len - 1, 1, num_dofs),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float32,
        )

        _, _, _, dummy_info = env.backend.reset(
            rngs.env(),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
        )

        # 3 state observations, 3 history observations for positions and velocities, 2 history observations for control
        # signals.
        robot_obs_per_dof = 3 + num_history_states * 3 - 1
        assert env.observation_spaces.shape == (1, (robot_obs_per_dof + 1) * num_dofs)
        pos_indices = [i * robot_obs_per_dof for i in range(num_dofs)]
        vel_indices = [i * robot_obs_per_dof + 1 for i in range(num_dofs)]
        ctrl_indices = [i * robot_obs_per_dof + 2 for i in range(num_dofs)]
        goal_indices = [i + 4 * robot_obs_per_dof for i in range(num_dofs)]
        vel_context_indices = jnp.array(
            [
                [i * robot_obs_per_dof + offset for offset in np.array([6, 7, 8])]
                for i in range(num_dofs)
            ]
        )
        control_context_indices = jnp.array(
            [
                [i * robot_obs_per_dof + offset for offset in np.array([9, 10])]
                for i in range(num_dofs)
            ]
        )

        observations = []
        (obs, _), state = env.reset(rngs.env())
        reset_ctrl = obs[:, ctrl_indices]

        controls = [reset_ctrl]
        for act in actions:
            controls.append(
                np.clip(controls[-1] + env.max_control_diff * act, -1.0, 1.0)
            )
        controls = np.stack(controls[1:], axis=0)

        for act, pos, vel in zip(actions, positions, velocities):
            # Hack to make step return position pos
            env.backend._step = lambda _rng, _buffers, _state, **_: (
                pos,
                {},
                _state,
                dummy_info,
            )
            # Changing env.backend._step does not trigger a recompile of env.backend.step, so calling env.backend.step()
            # will still use the old function --> Explicitly trigger the recompile by clearing JAX caches
            jax.clear_caches()
            (obs, _, _, _, _), state = env.step(act, state, rngs.env())
            observations.append(obs)

        last_pos = None
        for i, (obs, pos, vel, act) in enumerate(
            zip(observations, positions, velocities, actions)
        ):
            assert obs.shape == env.observation_spaces.shape
            assert jnp.all(obs[0, pos_indices] == pos)
            if last_pos is not None:
                assert obs[0, vel_indices] == pytest.approx(
                    (pos - last_pos)[0] / env.backend.dt
                )
            assert obs[:, ctrl_indices] == pytest.approx(controls[i])
            # Check that the goal positions for all dofs are non-zero (fails if one goal position is zero, but the
            # probability for that is basically zero)
            assert jnp.all(obs[0, goal_indices] != 0.0)
            # The goal position is constant throughout the episode
            assert jnp.all(obs[0, goal_indices] == observations[0][0, goal_indices])
            # Check that all valid history entries are zero (valid in the sense that enough steps have passed to
            # populate these fields)
            nonzero_control_context_indices = control_context_indices[
                :,
                range(
                    num_history_states - i,
                    num_history_states,
                ),
            ].flatten()
            assert jnp.all(obs[0, nonzero_control_context_indices] != 0.0)
            nonzero_vel_context_indices = vel_context_indices[
                :,
                range(
                    num_history_states - i - 1,
                    num_history_states,
                ),
            ].flatten()
            assert jnp.all(obs[0, nonzero_vel_context_indices] != 0.0)
            repeat_ctrl_context_indices = control_context_indices[
                :, range(num_history_states - i - 1)
            ].flatten()
            assert jnp.all(
                obs[0, repeat_ctrl_context_indices]
                == obs[0, control_context_indices[:, -i - 1]].repeat(
                    num_history_states - i - 1
                )
            )
            zero_vel_context_indices = vel_context_indices[
                :, range(num_history_states - i - 1)
            ].flatten()
            assert jnp.all(obs[0, zero_vel_context_indices] == 0.0)
            last_pos = pos
    finally:
        env.close()


def test_observation_bounds():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    env = PamyReacherEnv(backend, num_instances=1)
    try:
        (obs, _), state = env.reset(rngs.env())
        observations = [obs]
        for _ in range(500):
            (obs, _, _, _, _), state = env.step(
                jnp.array(env.action_spaces.sample()), state, rngs.env()
            )
            observations.append(np.array(obs))
        observations = np.concatenate(observations)
        assert np.all(observations <= env.observation_space.high)
        assert np.all(observations >= env.observation_space.low)
    finally:
        env.close()


@pytest.mark.parametrize(
    "action_repeat",
    [1, 2, 5],
)
def test_termination(action_repeat):
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    episode_length_s = 0.5
    env = PamyReacherEnv(
        backend,
        num_instances=2,
        episode_length_s=episode_length_s,
        action_repeat=action_repeat,
    )

    try:
        _, state = env.reset(rngs.env())
        done = jnp.array([False, False])
        num_steps = 0
        terminated = None
        truncated = None
        while not jnp.any(done):
            (_, _, terminated, truncated, info), state = env.step(
                jnp.array(env.action_spaces.sample()), state, rngs.env()
            )
            num_steps += 1
            done = jnp.logical_or(terminated, truncated)

        assert jnp.all(truncated) and not jnp.any(terminated)
        assert env.max_episode_length == math.ceil(episode_length_s / backend.dt)
        assert num_steps == env.max_episode_length // action_repeat

    finally:
        env.close()


def test_rewards():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    episode_length_s = 0.5
    env = PamyReacherEnv(backend, num_instances=10, episode_length_s=episode_length_s)

    try:
        _, dummy_obj_data, _, dummy_info = env.backend.reset(
            rngs.env(),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
        )

        _, state = env.reset(rngs.env())

        # Same goal, different positions
        goal_pos = jnp.stack([env.workspace_limits_lower] * env.num_instances)
        assert goal_pos.shape == state.task.goal_pos.shape
        state.task.goal_pos = goal_pos
        positions = jnp.linspace(
            env.workspace_limits_lower,
            env.workspace_limits_upper,
            env.num_instances,
        )
        # Hack to make step() return custom positions
        env.backend._step = lambda _rng, _buffers, _state, **kwargs: (
            positions,
            dummy_obj_data,
            _state,
            dummy_info,
        )
        (_, rewards_same_goal, _, _, _), state = env.step(
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32), state, rngs.env()
        )
        assert jnp.all(jnp.logical_and(-6 <= rewards_same_goal, rewards_same_goal <= 0))
        assert jnp.all(rewards_same_goal[1:] < rewards_same_goal[:-1])
        assert rewards_same_goal[0].item() == pytest.approx(0.0)
        assert rewards_same_goal[-1].item() == pytest.approx(-4.0)

        # Different goals, same position
        goal_pos = jnp.linspace(
            np.array(env.workspace_limits_lower),
            np.array(env.workspace_limits_upper),
            env.num_instances,
        )
        assert goal_pos.shape == state.task.goal_pos.shape
        state.task.goal_pos = goal_pos
        positions = jnp.stack([env.workspace_limits_upper] * env.num_instances)
        # Hack to make step() return specified positions
        env.backend._step = lambda _rng, _buffers, _state, **kwargs: (
            positions,
            dummy_obj_data,
            _state,
            dummy_info,
        )
        # Changing env.backend._step does not trigger a recompile of env.backend.step, so calling env.backend.step()
        # will still use the old function --> Explicitly trigger the recompile by clearing JAX caches
        jax.clear_caches()
        (_, rewards_diff_goals, _, _, _), state = env.step(
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32), state, rngs.env()
        )
        assert np.array(np.flip(rewards_diff_goals, 0)) == pytest.approx(
            np.array(rewards_same_goal)
        )
    finally:
        env.close()


def test_env_repeatability():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    env = PamyReacherEnv(backend, num_instances=2)

    try:
        action_traj = [
            jnp.array(env.action_spaces.sample())
            for _ in range(env.max_episode_length - 1)
        ]
        (obs, _), state = env.reset(rngs.env())
        obs_orig = [np.array(obs)]
        rew_orig = []
        term_orig = []
        trunc_orig = []
        for action in action_traj:
            (obs, rew, term, trunc, _), state = env.step(action, state, rngs.env())
            obs_orig.append(np.array(obs))
            rew_orig.append(np.array(rew))
            term_orig.append(np.array(term))
            trunc_orig.append(np.array(trunc))

        env.close()

        rngs = nnx.Rngs(params=13, backend=23, env=42)
        backend = MjxBackend(
            "racket",
            actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
            actuator_net_name=ACTUATOR_NET_NAME,
        )
        env = PamyReacherEnv(backend, num_instances=2)

        (obs, _), state = env.reset(rngs.env())
        obs_repeat = [np.array(obs)]
        rew_repeat = []
        term_repeat = []
        trunc_repeat = []
        for action in action_traj:
            (obs, rew, term, trunc, _), state = env.step(action, state, rngs.env())
            obs_repeat.append(np.array(obs))
            rew_repeat.append(np.array(rew))
            term_repeat.append(np.array(term))
            trunc_repeat.append(np.array(trunc))

        assert np.stack(obs_repeat) == pytest.approx(np.stack(obs_orig))
        assert np.stack(rew_repeat) == pytest.approx(np.stack(rew_orig))
        assert np.all(np.stack(term_repeat) == np.stack(term_orig))
        assert np.all(np.stack(trunc_repeat) == np.stack(trunc_orig))
    finally:
        env.close()


def test_reset_repeatability():
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    env = PamyReacherEnv(backend, num_instances=2)

    try:
        action_traj = [
            jnp.array(env.action_spaces.sample())
            for _ in range(env.max_episode_length - 1)
        ]
        rngs = nnx.Rngs(params=13, backend=23, env=42)
        env.rngs = rngs
        env.backend.rngs = rngs
        (obs, _), state = env.reset(rngs.env())
        obs_orig = [obs]
        rew_orig = []
        term_orig = []
        trunc_orig = []
        for action in action_traj:
            (obs, rew, term, trunc, info), state = env.step(action, state, rngs.env())
            obs_orig.append(np.array(obs))
            rew_orig.append(np.array(rew))
            term_orig.append(np.array(term))
            trunc_orig.append(np.array(trunc))

        rngs = nnx.Rngs(params=13, backend=23, env=42)
        env.rngs = rngs
        env.backend.rngs = rngs
        (obs, _), state = env.reset(rngs.env())
        obs_repeat = [obs]
        rew_repeat = []
        term_repeat = []
        trunc_repeat = []
        for action in action_traj:
            (obs, rew, term, trunc, info), state = env.step(action, state, rngs.env())
            obs_repeat.append(np.array(obs))
            rew_repeat.append(np.array(rew))
            term_repeat.append(np.array(term))
            trunc_repeat.append(np.array(trunc))

        assert np.stack(obs_repeat) == pytest.approx(np.stack(obs_orig))
        assert np.stack(rew_repeat) == pytest.approx(np.stack(rew_orig))
        assert np.all(np.stack(term_repeat) == np.stack(term_orig))
        assert np.all(np.stack(trunc_repeat) == np.stack(trunc_orig))
    finally:
        env.close()


def test_compare_an_input_and_observations():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    num_history_states = 3
    an_cfg = get_an_training_config(
        ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME
    ).actuator_net
    env = PamyReacherEnv(
        backend,
        num_instances=1,
        num_history_states=an_cfg.num_history_states,
        step_size=an_cfg.step_size,
        delta_history=an_cfg.delta_history,
    )

    try:
        num_dofs = 4
        robot_obs_per_dof = 3 + num_history_states * 3 - 1
        assert env.observation_spaces.shape == (1, (robot_obs_per_dof + 1) * num_dofs)
        pos_indices_obs = [i * robot_obs_per_dof for i in range(num_dofs)]
        vel_indices_obs = [i * robot_obs_per_dof + 1 for i in range(num_dofs)]
        ctrl_indices_obs = [i * robot_obs_per_dof + 2 for i in range(num_dofs)]
        pos_context_indices_obs = jnp.array(
            [
                [i * robot_obs_per_dof + offset for offset in np.array([3, 4, 5])]
                for i in range(num_dofs)
            ]
        )
        vel_context_indices_obs = jnp.array(
            [
                [i * robot_obs_per_dof + offset for offset in np.array([6, 7, 8])]
                for i in range(num_dofs)
            ]
        )
        ctrl_context_indices_obs = jnp.array(
            [
                [i * robot_obs_per_dof + offset for offset in np.array([9, 10])]
                for i in range(num_dofs)
            ]
        )
        an_inputs_per_dof = (num_history_states + 1) * 3
        ctrl_indices_an = [i * an_inputs_per_dof + 3 for i in range(num_dofs)]
        pos_indices_an = [i * an_inputs_per_dof + 7 for i in range(num_dofs)]
        vel_indices_an = [i * an_inputs_per_dof + 11 for i in range(num_dofs)]
        ctrl_context_indices_an = [
            [i * an_inputs_per_dof + offset for offset in [0, 1, 2]]
            for i in range(num_dofs)
        ]
        pos_context_indices_an = [
            [i * an_inputs_per_dof + offset for offset in [4, 5, 6]]
            for i in range(num_dofs)
        ]
        vel_context_indices_an = [
            [i * an_inputs_per_dof + offset for offset in [8, 9, 10]]
            for i in range(num_dofs)
        ]

        (obs, _), state = env.reset(rngs.env())
        for i in range(10):
            # update observations
            action = jnp.array(env.action_spaces.sample())
            last_obs = np.array(obs)
            (obs, _, _, _, info), state = env.step(action, state, rngs.env())
            an_input = np.array(info["backend"]["actuator_network"]["input"])

            assert np.all(last_obs[:, pos_indices_obs] == an_input[:, pos_indices_an])
            assert np.all(last_obs[:, vel_indices_obs] == an_input[:, vel_indices_an])
            if i > 0:
                if an_cfg.delta_history:
                    assert last_obs[:, ctrl_context_indices_obs] + last_obs[
                        :, ctrl_indices_obs, None
                    ] == pytest.approx(
                        an_input[:, ctrl_context_indices_an]
                        + an_input[:, ctrl_indices_an, None],
                        abs=1e-6,
                    )
                else:
                    assert np.all(
                        last_obs[:, ctrl_context_indices_obs]
                        == an_input[:, ctrl_context_indices_an]
                    )
            else:
                # Directly after the reset, the control context in the observations is 0, and in the AN input it is set
                # to the first control.
                assert np.all(last_obs[:, ctrl_context_indices_obs] == 0.0)
                if an_cfg.delta_history:
                    assert np.all(an_input[:, ctrl_context_indices_an] == 0.0)
                else:
                    assert np.all(
                        an_input[:, ctrl_context_indices_an]
                        == an_input[:, ctrl_indices_an][:, :, None]
                    )
            assert np.all(
                last_obs[:, vel_context_indices_obs]
                == an_input[:, vel_context_indices_an]
            )
            assert np.all(
                last_obs[:, pos_context_indices_obs]
                == an_input[:, pos_context_indices_an]
            )

    finally:
        env.close()


def test_reset_overwrite_single():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    env = PamyReacherEnv(backend, num_instances=5)
    try:
        _, state = env.reset(rngs.env())
        overwrite_pos = jnp.array(
            [
                [0.01, 0.02, 0.03, 0.04],
                [0.05, 0.06, 0.07, 0.08],
                [0.09, 0.010, 0.011, 0.012],
            ],
            dtype=jnp.float32,
        )
        overwrite_prev_pos = jnp.array(
            [
                [1.01, 1.02, 1.03, 1.04],
                [1.05, 1.06, 1.07, 1.08],
                [1.09, 1.010, 1.011, 1.012],
            ],
            dtype=jnp.float32,
        )
        _, _, backend_state, _ = env.backend.reset(
            rngs.backend(),
            jnp.zeros_like(overwrite_pos),
            overwrite_pos,
            overwrite_prev_pos,
        )

        assert jnp.all(backend_state.mjx_data.qpos == overwrite_pos)
        velocity = (overwrite_pos - overwrite_prev_pos) / env.backend.dt
        assert jnp.all(backend_state.mjx_data.qvel == velocity)

        _, state = env.reset(rngs.env())
        for i in range(len(overwrite_pos)):
            assert jnp.all(
                jnp.any(state.backend.mjx_data.qpos != overwrite_pos[i][None], axis=1)
            )
            assert jnp.all(
                jnp.any(state.backend.mjx_data.qvel != velocity[i][None], axis=1)
            )
    finally:
        env.close()


def test_reset_overwrite_hist():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    num_history_states = 3
    step_size = 3
    num_instances = 5

    hist_len = num_history_states * step_size + 1
    overwrite_pos_hist = jax.random.uniform(
        rngs.env(), (num_instances, hist_len, 4), dtype=jnp.float32
    )
    overwrite_ctrl_hist = jax.random.uniform(
        rngs.env(), (num_instances, hist_len - 1, 4), dtype=jnp.float32
    )
    buffers, _, state, _ = backend.reset_with_history(
        rngs.backend(), overwrite_pos_hist, overwrite_ctrl_hist
    )

    assert jnp.all(buffers.positions == overwrite_pos_hist)
    assert jnp.all(buffers.controls[:, 1:] == overwrite_ctrl_hist)


def test_joint_limits_loss():
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )
    env = PamyReacherEnv(backend, num_instances=8)

    try:
        _, dummy_obj_data, _, dummy_info = env.backend.reset(
            rngs.env(),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32),
        )

        _, state = env.reset(rngs.env())
        ws_lower = env.workspace_limits_lower
        ws_upper = env.workspace_limits_upper
        positions = jax.random.uniform(
            rngs.env(),
            (env.num_instances, 4),
            minval=ws_lower,
            maxval=ws_upper,
            dtype=jnp.float32,
        )
        lower = env.joint_limits_lower
        upper = env.joint_limits_upper
        max_loss = 1.0
        positions = positions.at[0, 0].set(lower[0])
        positions = positions.at[1, 1].set(upper[1])
        positions = positions.at[2, 2].set(lower[2])
        positions = positions.at[3, 3].set(upper[3])
        positions = positions.at[5, :].set(
            jnp.array([upper[0], lower[1], upper[2], lower[3]])
        )
        positions = positions.at[6, 0].set(
            lower[1] + 0.5 * env.joint_limits_penalty_dist
        )
        positions = positions.at[7, 1].set(
            upper[2] - 0.5 * env.joint_limits_penalty_dist
        )
        velocities = jnp.zeros((env.num_instances, 4), dtype=jnp.float32)

        # Hack to make step() return custom positions
        env.backend._step = lambda _rng, _buffers, _state, **kwargs: (
            positions,
            dummy_obj_data,
            _state,
            dummy_info,
        )
        (_, rewards, _, _, _), state = env.step(
            jnp.zeros((env.num_instances, 4), dtype=jnp.float32), state, rngs.env()
        )

        expected_joint_limit_loss = np.array(
            [
                max_loss,
                max_loss,
                max_loss,
                max_loss,
                0.0,
                4.0 * max_loss,
                0.5 * max_loss,
                0.5 * max_loss,
            ]
        )

        assert state.joint_limit_loss == pytest.approx(expected_joint_limit_loss)
        assert np.all(np.array(rewards) < -expected_joint_limit_loss)
    finally:
        env.close()


def test_reset_incomplete_history():
    dt = 0.001
    rngs = nnx.Rngs(params=13, backend=23, env=42)
    backend = MjxBackend(
        "racket",
        actuator_net_exp_name=ACTUATOR_NET_EXP_NAME,
        actuator_net_name=ACTUATOR_NET_NAME,
    )

    an_cfg = get_an_training_config(
        ACTUATOR_NET_EXP_NAME, ACTUATOR_NET_NAME
    ).actuator_net
    env = PamyReacherEnv(backend, num_instances=8, delta_history=an_cfg.delta_history)

    try:
        _, state = env.reset(rngs.env())
        # After reset the buffer holds the previous reset position repeated for the whole
        # history followed by the final reset position, i.e. only the last two positions
        # carry information. Stepping with the (constant) reset control must therefore
        # yield the same AN input as padding that (prev, final) position pair and the
        # single control to the full history length.
        position_pair = state.buffers.positions[:, -2:]
        control = state.buffers.controls[:, -1]
        _, _, _, info = env.backend.step(
            rngs.env(), control, state.buffers, state.backend
        )
        an_input_env = np.array(info["actuator_network"]["input"])

        an_input_pack = np.array(
            pack_an_input(
                position_pair,
                None,
                control[:, None],
                backend.actuator_net_cfg,
                dt,
                pad_incomplete_hist=True,
            )
        )
        assert np.all(an_input_env == an_input_pack)
    finally:
        env.close()
