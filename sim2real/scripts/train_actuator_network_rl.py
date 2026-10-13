import logging
import pickle
import subprocess
from pathlib import Path

import hydra
import jax.numpy as jnp
import numpy as np
import omegaconf
import skrl.agents.jax
import skrl.envs.wrappers.jax
import skrl.trainers.jax
from flax import nnx

from sim2real.actuator_network.actuator_net_env import ActuatorNetEnv
from sim2real.actuator_network.dataset import Dataset
from sim2real.actuator_network.models import convert_actuator_net_rl
from sim2real.actuator_network.simulation_accuracy_evaluator import (
    SimulationAccuracyEvaluator,
)
from sim2real.data_processing.process_data import process_data
from sim2real.rl_tasks.agent import create_agent
from sim2real.rl_tasks.custom_trainer import CustomSequentialTrainer
from sim2real.rl_tasks.physics_backend.mjx_backend import MjxBackend
from sim2real.rl_tasks.skrl_env_wrapper import SkrlEnvWrapper
from sim2real.scripts.train_actuator_network import split_datasets

skrl.config.jax.backend = "jax"
omegaconf.OmegaConf.register_new_resolver("eval", eval)

logger = logging.getLogger(__name__)


def train_actuator_network_rl(cfg: omegaconf.DictConfig) -> None:
    if cfg["seed"] is not None:
        seed = cfg["seed"]
    else:
        seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)

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
        plot_visualizations=False,
    )

    dt = 1 / cfg.target_frequency

    dataset = Dataset(
        positions,
        positions_raw,
        observed_pressures,
        controls,
        torques,
    )
    train_data, val_data, train_data_stats = split_datasets(
        cfg,
        dataset,
        dt,
        train_split=0.8,
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
            plot_visualizations=False,
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

    backend = MjxBackend(
        context=cfg.end_effector,
        actuator_net_cfg=cfg.actuator_net,
        actuator_net_timestep=dt,
    )
    env = ActuatorNetEnv(
        backend,
        train_data=train_data,
        actuator_net_cfg=cfg.actuator_net,
        **cfg.env,
    )

    rngs = nnx.Rngs(env=seed, backend=seed + 1)
    env = SkrlEnvWrapper(env, rngs)

    agent = create_agent(
        cfg.agent, env.observation_space, env.action_space, env.num_instances
    )

    sim_acc_eval = SimulationAccuracyEvaluator(
        rngs,
        train_data,
        val_data,
        extra_eval_data,
        dt,
        cfg.actuator_net,
        cfg.end_effector,
        num_train_episodes=cfg.sim_acc_eval_episodes,
        num_val_episodes=cfg.sim_acc_eval_episodes,
        num_steps_per_episode=max(cfg.sim_acc_eval_steps_to_log),
    )

    def sim_acc_eval_callback(
        agent: skrl.agents.jax.Agent, env: skrl.envs.wrappers.jax.Wrapper, timestep: int
    ):
        if timestep % cfg.sim_acc_eval_interval == 0:
            actuator_ensemble = convert_actuator_net_rl(agent, cfg)
            (
                (train_pos_errs, train_pos_errs_quant, _, _),
                (val_pos_errs, val_pos_errs_quant, _, _),
                (extra_eval_pos_errs, extra_eval_pos_errs_quant, _, _),
            ) = sim_acc_eval.evaluate(actuator_ensemble)
            subsampling_factor = round(dt / 0.001)
            for name, errors, quant in [
                ("train/position", train_pos_errs, False),
                ("val/position", val_pos_errs, False),
                ("extra/position", extra_eval_pos_errs, False),
                ("train/position", train_pos_errs_quant, True),
                ("val/position", val_pos_errs_quant, True),
                ("extra/position", extra_eval_pos_errs_quant, True),
            ]:
                suffix = "_quant" if quant else ""
                if errors is not None:
                    for step in cfg.sim_acc_eval_steps_to_log:
                        if errors.shape[1] >= step:
                            agent.track_data(
                                f"sim_error_deg_avg{suffix}/{name}/step{step * subsampling_factor:04d}",
                                jnp.abs(errors[:, step - 1]).mean().item()
                                * 180.0
                                / np.pi,
                            )
                    for dof in range(4):
                        for step in cfg.sim_acc_eval_steps_to_log:
                            if errors.shape[1] >= step:
                                agent.track_data(
                                    f"sim_error_deg_joints{suffix}/{name}/step{step * subsampling_factor:04d}/dof{dof}",
                                    jnp.abs(errors[:, step - 1, dof]).mean().item()
                                    * 180.0
                                    / np.pi,
                                )
            agent.write_tracking_data(timestep=timestep, timesteps=None)

    cfg_trainer = {
        "timesteps": cfg.num_timesteps,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": True,
    }
    trainer = CustomSequentialTrainer(
        env=env,
        agents=agent,
        cfg=cfg_trainer,
        post_interaction_callback=sim_acc_eval_callback,
    )
    trainer.train()
    if agent.write_interval > 0:
        # The writer thread is a daemon, so pending events would be lost when the
        # process exits without flushing them first.
        agent.writer.flush()


@hydra.main(
    version_base=None, config_path="config", config_name="train_actuator_network_rl"
)
def main(cfg: omegaconf.DictConfig) -> None:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        logger.info(f"Git commit: {sha}")
    except subprocess.CalledProcessError:
        logger.info("Not in a git repository")
    cfg.actuator_net.ensemble_size = 1
    train_actuator_network_rl(cfg)


if __name__ == "__main__":
    main()
