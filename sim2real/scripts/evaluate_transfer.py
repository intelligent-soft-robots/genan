import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import skrl
from flax import nnx
from skrl.agents.jax.ppo import PPO
from tqdm import trange

from sim2real.rl_tasks.agent import load_agent
from sim2real.rl_tasks.pamy_ball_in_cup.pamy_ball_in_cup import PamyBallInCupEnv
from sim2real.rl_tasks.pamy_env import PamyEnv
from sim2real.rl_tasks.pamy_reacher.pamy_reacher import PamyReacherEnv
from sim2real.rl_tasks.pamy_table_tennis.pamy_table_tennis import PamyTableTennisEnv
from sim2real.rl_tasks.physics_backend.physics_backend import ObjData
from sim2real.scripts.train_rl_agent import create_env
from sim2real.util.config_utils import get_an_training_config

logger = logging.getLogger(__name__)

try:
    from sim2real.rl_tasks.physics_backend.o80_backend import O80Backend
except ModuleNotFoundError:
    logger.warning("O80 backend not found.")
    O80Backend = None


skrl.config.jax.backend = "jax"


def evaluate_agent(
    key: jax.random.PRNGKey,
    agent: PPO,
    env: PamyEnv,
    num_episodes: int,
):
    @jax.jit
    def agent_act(obs: jnp.ndarray) -> jnp.ndarray:
        return agent.policy.apply(
            agent.policy.state_dict.params,
            {"observations": agent._observation_preprocessor(obs)},
            "policy",
        )[0]

    @jax.jit
    def eval_episode(key: jax.random.PRNGKey, dry_run: jnp.ndarray):
        (obs, info), state = env.reset(key, dry_run)
        max_episode_length = env.max_episode_length // env.action_repeat
        ep_obs = jnp.full(
            (
                env.num_instances,
                max_episode_length + 1,
                env.observation_space.shape[0],
            ),
            jnp.nan,
        )
        ep_obs = ep_obs.at[:, 0].set(obs)
        ep_acts = jnp.full(
            (
                env.num_instances,
                max_episode_length,
                env.action_space.shape[0],
            ),
            jnp.nan,
        )
        ep_poss = jnp.full((env.num_instances, max_episode_length + 1, 4), jnp.nan)
        ep_poss = ep_poss.at[:, 0].set(info["position"])
        ep_ctrls = jnp.full((env.num_instances, max_episode_length + 1, 4), jnp.nan)
        ep_ctrls = ep_ctrls.at[:, 0].set(info["control"])
        obj_datas = jax.tree.map(
            lambda x: jnp.concatenate(
                (
                    x[:, None],
                    jnp.full(
                        (x.shape[0], max_episode_length, *x.shape[1:]),
                        jnp.nan,
                        dtype=x.dtype,
                    ),
                ),
                axis=1,
            ),
            state.obj_data,
        )
        backend_infos = jax.tree.map(
            lambda x: jnp.concatenate(
                (
                    x[..., None, :],
                    jnp.full(
                        (*x.shape[:-1], max_episode_length, x.shape[-1]),
                        jnp.nan,
                        dtype=x.dtype,
                    ),
                ),
                axis=-2,
            ),
            info["backend"],
        )

        def step(
            carry: Tuple[
                jnp.ndarray,
                jnp.ndarray,
                Any,
                jax.random.PRNGKey,
                jnp.ndarray,
                jnp.ndarray,
                Dict[str, ObjData],
                jnp.ndarray,
                jnp.ndarray,
                Dict[str, Any],
                jnp.ndarray,
                Dict[str, Any],
            ],
        ):
            (
                done,
                _obs,
                _state,
                _key,
                _ep_obs,
                _ep_acts,
                _obj_datas,
                _ep_poss,
                _ep_ctrls,
                _backend_infos,
                _sum_rewards,
                _,
            ) = carry
            _key, _key_step = jax.random.split(_key)
            act = agent_act(_obs)
            # TODO: Remove this hack!
            if isinstance(env, PamyTableTennisEnv):
                act = jnp.where(_state.backend.step <= 545, act, jnp.zeros_like(act))
            (_obs, rew, term, trunc, _info), _state = env.step(
                act, _state, _key_step, dry_run
            )
            done = jnp.logical_or(jnp.all(term), jnp.all(trunc))
            s = _state.backend.step // env.action_repeat
            _ep_obs = _ep_obs.at[:, s].set(_obs)
            _ep_acts = _ep_acts.at[:, s].set(act)
            _backend_infos = jax.tree.map(
                lambda _ep, _cur: _ep.at[..., s, :].set(_cur),
                _backend_infos,
                _info["backend"],
            )
            _obj_datas = jax.tree.map(
                lambda _ep, _cur: _ep.at[:, s].set(_cur),
                _obj_datas,
                _state.obj_data,
            )
            _ep_poss = _ep_poss.at[:, s].set(_info["position"])
            _ep_ctrls = _ep_ctrls.at[:, s].set(_info["control"])
            return (
                done,
                _obs,
                _state,
                _key,
                _ep_obs,
                _ep_acts,
                _obj_datas,
                _ep_poss,
                _ep_ctrls,
                _backend_infos,
                _sum_rewards + rew,
                _info["episode"],
            )

        (
            _,
            _,
            _,
            _,
            ep_obs,
            ep_acts,
            obj_datas,
            ep_poss,
            ep_ctrls,
            backend_infos,
            sum_rewards,
            ep_info,
        ) = flax.nnx.while_loop(
            lambda carry: jnp.logical_not(carry[0]),
            step,
            (
                jnp.array(False),
                obs,
                state,
                key,
                ep_obs,
                ep_acts,
                obj_datas,
                ep_poss,
                ep_ctrls,
                backend_infos,
                jnp.zeros(env.num_instances, dtype=jnp.float32),
                info["episode"],
            ),
        )
        return (
            ep_obs,
            ep_acts,
            obj_datas,
            ep_poss,
            ep_ctrls,
            backend_infos,
            sum_rewards,
            ep_info,
        )

    episode_rewards = []
    episode_observations = []
    episode_positions = []
    episode_controls = []
    episode_infos = []
    key, key_ep = jax.random.split(key)
    eval_episode(key_ep, dry_run=jnp.array(True))
    for i in trange(num_episodes):
        key, key_ep = jax.random.split(key)
        (
            observations,
            actions,
            obj_datas,
            positions,
            controls,
            backend_infos,
            sum_rewards,
            episode_info,
        ) = eval_episode(key_ep, dry_run=jnp.array(False))
        if isinstance(env, PamyReacherEnv):
            task = "reacher"
        elif isinstance(env, PamyBallInCupEnv):
            task = "ball_in_cup"
        else:
            task = "table_tennis"
        data = {"task": task}
        if O80Backend is not None and isinstance(env.backend, O80Backend):
            init_o80_iteration = backend_infos["iteration"][0, 0].item()
        else:
            init_o80_iteration = None
        if init_o80_iteration is not None:
            o80_observations = env.backend.robot_frontend.get_observations_since(
                init_o80_iteration
            )
            data["o80_iterations"] = np.array(
                [o.get_iteration() for o in o80_observations]
            )
            data["o80_timestamps"] = np.array(
                [env.backend._extract_timestamp(o) for o in o80_observations]
            )
            data["o80_positions"] = np.stack(
                [o.get_positions() for o in o80_observations]
            )
            data["o80_velocities"] = np.stack(
                [o.get_velocities() for o in o80_observations]
            )
            data["o80_desired_pressures"] = np.stack(
                [o.get_desired_pressures() for o in o80_observations]
            )
            data["o80_observed_pressures"] = np.stack(
                [o.get_observed_pressures() for o in o80_observations]
            )

        observations = np.array(observations)
        actions = np.array(actions)
        positions = np.array(positions)
        controls = np.array(controls)
        backend_infos = jax.tree.map(np.array, backend_infos)
        episode_info = jax.tree.map(np.array, episode_info)

        episode_rewards.append(jnp.mean(sum_rewards).item())
        episode_observations.append(observations)
        episode_positions.append(positions)
        episode_controls.append(controls)
        episode_infos.append(episode_info)

        ep_out_file = f"ep_{i}.npz"
        data["observations"] = observations
        data["actions"] = actions
        data["sum_rewards"] = np.array(sum_rewards)
        data["backend_infos"] = backend_infos
        data["episode_info"] = episode_info
        data["positions"] = positions
        data["controls"] = controls
        data["object_datas"] = jax.tree.map(lambda x: np.array(x), obj_datas)
        np.savez(ep_out_file, **data)

        print(f"episode_rewards: {np.mean(episode_rewards)}")
        if isinstance(env, PamyReacherEnv):
            success_rate_str = (
                f"success_rate: "
                f"{np.mean([inf['target_dist/success_rate_norm_1steps_dist0.1'] for inf in episode_infos]) * 100.0:.2f}%"
            )
        elif isinstance(env, PamyBallInCupEnv):
            success_rate_str = (
                f"success_rate: "
                f"{np.mean([inf['ball_in_cup/success_rate'] for inf in episode_infos]) * 100.0:.2f}%"
            )
        else:
            success_rate_str = (
                f"success_rate: "
                f"{np.mean([inf['table_tennis/success_rate_dist_0.685'] for inf in episode_infos]) * 100.0:.2f}%"
            )
        print(success_rate_str)
        logger.info(success_rate_str)


@hydra.main(version_base=None, config_path="config", config_name="evaluate_transfer")
def main(cfg: omegaconf.DictConfig):
    if cfg.force_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])

    train_logs_path = Path(__file__).parents[2] / "logs" / "train"
    exps_path = Path(__file__).parents[2] / "exps"
    training_logs_path = train_logs_path / cfg.train_logs

    train_cfg = omegaconf.OmegaConf.load(
        training_logs_path / "train_rl_agent_config" / "config.yaml"
    )

    an_exp_name = train_cfg.backend.actuator_net_exp_name
    an_name = train_cfg.backend.actuator_net_name
    an_cfg = get_an_training_config(an_exp_name, an_name).actuator_net

    data_collection_cfg = omegaconf.OmegaConf.load(
        exps_path / an_exp_name / "collect_data_config" / "config.yaml"
    )

    # TODO: Find a more elegant solution for this
    train_cfg.env.num_history_states = an_cfg.num_history_states
    train_cfg.env.step_size = an_cfg.step_size
    train_cfg.env.delta_history = an_cfg.delta_history

    for k, v in train_cfg.backend.items():
        if k in cfg.backend and k != "_target_" and cfg.backend[k] != v:
            logger.warning(
                f"Value for parameter {k} differs from training backend. "
                f"Training: {train_cfg.backend[k]}, evaluation: {cfg.backend[k]}."
            )

    hist_len = an_cfg.num_history_states * an_cfg.step_size + 1
    if "history_length" in cfg.backend:
        cfg.backend.history_length = hist_len
    if "history_length" in train_cfg.backend:
        train_cfg.backend.history_length = hist_len
    if "ndp" in cfg.backend:
        cfg.backend.ndp = data_collection_cfg.ndp
    if "terminate_on_success" in train_cfg.env:
        train_cfg.env.terminate_on_success = cfg.overwrite_terminate_on_success
    if (
        O80Backend is not None
        and hydra.utils.get_class(cfg.backend._target_) == O80Backend
    ):
        train_cfg.env.num_instances = 1
    cfg.backend.context = train_cfg.backend.context
    # TODO: Find a better solution to record the whole tt ball trajectory
    if hydra.utils.get_class(train_cfg.env._target_) is PamyTableTennisEnv:
        train_cfg.env.episode_length_s = 2.0

    train_cfg.backend = cfg.backend

    env = create_env(train_cfg)

    with (exps_path / an_exp_name / "robot_config.json").open("r") as f:
        data_robot_config = json.load(f)
    if hasattr(env.backend, "cocontraction"):
        assert np.all(
            data_robot_config["min_pressures_ago"]
            == env.backend.cocontraction.min_pressures_ago_np
        ), (
            f"min_pressures_ago does not match the dataset, expected {data_robot_config['min_pressures_ago']}, "
            f"got {env.backend.cocontraction.min_pressures_ago_np}."
        )
        assert np.all(
            data_robot_config["min_pressures_antago"]
            == env.backend.cocontraction.min_pressures_antago_np
        ), (
            f"min_pressures_antago does not match the dataset, expected {data_robot_config['min_pressures_antago']}, "
            f"got {env.backend.cocontraction.min_pressures_antago}."
        )
        assert np.all(
            data_robot_config["max_pressures_ago"]
            == env.backend.cocontraction.max_pressures_ago_np
        ), (
            f"max_pressures_ago does not match the dataset, expected {data_robot_config['max_pressures_ago']}, "
            f"got {env.backend.cocontraction.max_pressures_ago}."
        )
        assert np.all(
            data_robot_config["max_pressures_antago"]
            == env.backend.cocontraction.max_pressures_antago_np
        ), (
            f"max_pressures_antago does not match the dataset, expected {data_robot_config['max_pressures_antago']}, "
            f"got {env.backend.cocontraction.max_pressures_antago}."
        )
    print("Environment created")

    agent = load_agent(
        (training_logs_path / "checkpoints" / cfg.checkpoint_name).with_suffix(
            ".pickle"
        ),
        train_cfg.agent,
        env.observation_space,
        env.action_space,
        env.num_instances,
        device="cpu" if cfg.force_cpu else None,
    )
    # Set to agent to eval mode
    agent.enable_training_mode(False)

    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))

    evaluate_agent(
        key,
        agent,
        env,
        cfg.num_episodes,
    )


if __name__ == "__main__":
    # run the main function
    main()
